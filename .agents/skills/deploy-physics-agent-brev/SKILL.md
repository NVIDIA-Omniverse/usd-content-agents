---
name: deploy-physics-agent-brev
description: Deploy or smoke-test the Physics Agent with Brev-hosted dependency endpoints. Use when the user asks to test physics agent on Brev, deploy physics agent with Brev, use RTX/L40S for rendering and A100/H100-grade GPU for VLM, run a Brev hybrid physics-agent test, recreate the Brev physics-agent deployment, or validate service-side refine with a registered provider. This workflow keeps the main pipeline local and uses Brev port-forwards to reach OVRTX and Qwen-family VLM endpoints.
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - brev
  - ovrtx
  - vlm
  - deployment
tools:
  - Shell
  - Docker
  - Python
  - curl
  - Filesystem
  - brev
  - ssh
compatibility: Requires Brev CLI access, a remote RTX/L40S-class render GPU for OVRTX, an A100/H100-class VLM endpoint, local port-forwards, and Physics Agent provider credentials.
---

# Deploy Physics Agent With Brev

## When to Use

Use the credit-conscious hybrid path:

- Local machine runs the physics-agent pipeline.
- Brev `wu-pa-render` runs standalone OVRTX on an RTX GPU. Prefer AWS
  `g7e.2xlarge` / RTX PRO Server 6000 for fast validation, or AWS
  `g6e.xlarge` / L40S when lower cost matters more than cold-start latency.
- Brev `wu-pa-vlm-a100` or `wu-pa-vlm-h100` runs an OpenAI-compatible Qwen
  endpoint. Use Denvr A100 80 GB for the lower-cost validated path, or
  Hyperstack H100 for the validated Qwen3.5 35B NIM path.
- Local port-forwards expose `RENDER_ENDPOINT=http://localhost:8001` and
  `PA_VLM_NIM_BASE_URL=http://localhost:8003/v1`.

## Limitations

- Keep provider credentials in `.env` or remote env files with restrictive
  permissions; never print or commit them.
- Prefer local port-forwards for this hybrid path; do not assume Brev
  instance-to-instance networking is available.
- Physics Agent service `/refine` requires the `tuning` dependencies, an
  OvPhysX runtime, and a registered chat/VLM provider configured through
  `PA_REFINE_BACKEND`, `PA_REFINE_MODEL`, and that provider's credential.
- Delete GPU nodes after validation unless the user asks to keep them.

## Prerequisites

- Brev CLI access and SSH-ready render and VLM nodes.
- Docker, NVIDIA Container Toolkit, and writable storage on each remote node.
- Local Physics Agent environment with provider credentials required by the
  selected config.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Create or reuse the Brev render node, then validate OVRTX health and render.
3. Create or reuse the VLM node, then validate model listing and image chat.
4. Set local Physics Agent render, VLM, and LLM environment variables.
5. Run the local pipeline with the forwarded endpoints, then clean up nodes and
   port-forwards.

## Credit Safety

Start with:

```bash
brev --version
brev healthcheck
brev ls --json
python scripts/brev_agent_services.py --service physics --preset hybrid --render-gpu-name RTX --vlm-gpu-name A100
```

Reuse existing instances when possible. Run `brev create --dry-run` before
real creation unless the user already confirmed the exact spend.

For service-on-Brev presets, the planner excludes local `.env` files during
worktree copy, then writes a minimal remote `.env` with generated endpoint/model
wiring and starts Docker Compose with `--env-file .env`. Edit that remote
`.env` before the Compose step when a generated comment asks for a real API key.

Useful dry-run candidates:

```bash
brev create wu-pa-render --dry-run --type g7e.2xlarge --min-disk 500
brev create wu-pa-render --dry-run --type g6e.xlarge --min-disk 500
brev create wu-pa-render --dry-run --type gpu-l40s-a.1gpu-8vcpu-32gb --min-disk 500
brev create wu-pa-vlm-a100 --dry-run --type denvr_A100_sxm4_80G --min-disk 500
brev create wu-pa-vlm-h100 --dry-run --type hyperstack_H100 --min-disk 500
```

## Render Node

Copy to a writable path; `/workspace` was not writable on observed images.

```bash
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .data \
  --exclude docs/metrics --exclude coverage.xml \
  --exclude .env --exclude '.env.*' --exclude '.env-*' \
  --exclude '.env *' --exclude .envrc \
  --exclude '*credentials*.json' --exclude '*private*key*' \
  --exclude '*.key' --exclude '*.pem' --exclude '*.p12' --exclude '*.p8' \
  --exclude id_rsa --exclude id_dsa --exclude id_ecdsa --exclude id_ed25519 \
  ./ wu-pa-render:~/world-understanding/
ssh wu-pa-render "cd ~/world-understanding && OVRTX_RENDER_MODE=pt OVRTX_NUM_SENSOR_UPDATES=1 OVRTX_DAEMON_RENDER_TIMEOUT=900 docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up -d --build"
```

If OVRTX logs `VkResult: ERROR_INCOMPATIBLE_DRIVER` or lacks
`libGLX_nvidia.so`, install the matching NVIDIA GL package and regenerate CDI:

```bash
ssh wu-pa-render "sudo apt-get install -y libnvidia-gl-580=<driver-version> vulkan-tools"
ssh wu-pa-render "sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo cp /etc/cdi/nvidia.yaml /var/run/cdi/nvidia.yaml"
ssh wu-pa-render "cd ~/world-understanding && OVRTX_RENDER_MODE=pt OVRTX_NUM_SENSOR_UPDATES=1 OVRTX_DAEMON_RENDER_TIMEOUT=900 docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up -d --force-recreate"
```

Then forward and verify:

```bash
brev port-forward wu-pa-render -p 8001:8001
curl -fsS http://localhost:8001/health
```

Do not use `rt2` for Physics Agent validation. For a first render-service
return-path smoke, it is acceptable to add `OVRTX_NUM_SENSOR_UPDATES=1` while
keeping `OVRTX_RENDER_MODE=pt`; restore the default before quality-sensitive
pipeline runs. Avoid very short cold-start render timeouts. A 900 second
timeout allowed warm-up and an OVRTX `/render` smoke to pass on validated AWS
RTX PRO Server 6000 and L40S nodes, but L40S can be noticeably slower.

## A100/H100 VLM Node

For the first lower-cost Brev model smoke test, serve `Qwen/Qwen2.5-VL-7B-Instruct` with
vLLM on Denvr `denvr_A100_sxm4_80G`. For the validated larger endpoint, use
the Hyperstack H100 path in `deploy-qwen-vlm-brev` with
`nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant`.
If the user asks for NVIDIA NIM, use `deploy-qwen-vlm-brev` and the validated
`nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant` path on A100 or H100. That
NIM requires request-level
`chat_template_kwargs: {"enable_thinking": false}` for direct answers.

```bash
brev exec wu-pa-vlm-a100 \
  "df -h / /home /mnt/* 2>/dev/null || true; \
   lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; \
   du -sh /var/lib/docker ~/.cache/huggingface 2>/dev/null || true; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
```

If the requested Brev disk size is not reflected in a writable filesystem,
delete the VM and try another provider/type. Do not promote a VLM VM type to
the preferred path until disk qualification, `/v1/models`, text chat, and image
chat all pass.

```bash
brev exec wu-pa-vlm-a100 \
  "sudo file -s /dev/vdc && sudo fdisk -l /dev/vdc"
brev exec wu-pa-vlm-a100 \
  "set -e; \
   sudo mkfs.ext4 -F /dev/vdc && \
   sudo mkdir -p /mnt/data && \
   sudo mount /dev/vdc /mnt/data && \
   sudo mkdir -p /mnt/data/docker /mnt/data/huggingface && \
   sudo chown -R \$(id -un):\$(id -gn) /mnt/data/huggingface && \
   sudo env DOCKER_DATA_ROOT=/mnt/data/docker python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path('/etc/docker/daemon.json')
text = path.read_text() if path.exists() else ''
data = json.loads(text) if text.strip() else {}
if not isinstance(data, dict):
    raise SystemExit(f'{path} must contain a JSON object')
data['data-root'] = os.environ['DOCKER_DATA_ROOT']
path.write_text(json.dumps(data, indent=2) + '\n')
PY
   sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && \
   sudo systemctl restart docker && \
   df -h /mnt/data && \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'"
brev exec wu-pa-vlm-a100 \
  "docker rm -f qwen-vlm >/dev/null 2>&1 || true; \
   docker run -d --name qwen-vlm --gpus all --ipc=host -p 8000:8000 \
     -v /mnt/data/huggingface:/root/.cache/huggingface \
     vllm/vllm-openai:v0.8.5.post1 \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
     --host 0.0.0.0 --port 8000 --dtype bfloat16 \
     --max-model-len 8192 --gpu-memory-utilization 0.80 \
     --max-num-seqs 1 --trust-remote-code \
     --limit-mm-per-prompt image=20 --enforce-eager"
brev exec wu-pa-vlm-a100 "docker logs --tail 100 qwen-vlm"
brev exec wu-pa-vlm-a100 "curl -fsS http://localhost:8000/v1/models"
brev port-forward wu-pa-vlm-a100 -p 8003:8000
curl -fsS http://localhost:8003/v1/models
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"What is the dominant color in this image? Answer with one word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="}}]}],"max_tokens":16,"temperature":0}'
```

## Local Physics Pipeline Environment

```bash
export RENDER_ENDPOINT=http://localhost:8001
export MA_RENDERING_USE_DATA_URI=true
export PA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export PA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_NIM_API_KEY=not-used
```

For the validated 35B NIM path, set `PA_VLM_MODEL` to
`qwen/qwen3.5-35b-a3b` and ensure the model config or request extra body passes
`chat_template_kwargs.enable_thinking=false`.
Use `PA_NIM_API_KEY=not-used` only for local no-auth VLM/LLM endpoints such as
a Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `PA_NIM_API_KEY` in `.env` instead of the
dummy key.

This does not require Brev-to-Brev networking because the local machine calls
both dependency endpoints.

## Validated H100 35B CLI Smoke

The validated Physics Agent hybrid smoke used:

- `wu-pa-render`: AWS `g7e.2xlarge` / RTX PRO Server 6000 running standalone
  OVRTX with `OVRTX_RENDER_MODE=pt`.
- `wu-pa-vlm-h100`: `hyperstack_H100` running
  `nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant`.
- Local port-forwards on separate ports, for example
  `RENDER_ENDPOINT=http://localhost:8011` and
  `PA_VLM_NIM_BASE_URL=http://localhost:8013/v1`.

When writing a short smoke config from `apps/physics_agent/configs/lightbulb.yaml`:

- Keep `optimize_usd.enabled: false` for the lightbulb smoke.
- For instanced CAD assets or assets that fail `apply_physics` on instance
  proxies, enable `optimize_usd` with
  `scene_optimizer_settings.enable_deinstance: true`. Also set
  `enable_split_meshes: true` when one combined mesh should become separate
  component predictions.
- Set both renderers to `backend: remote`.
- Use a valid render mode name such as `prim_only`; do not use
  `prim_only_original` unless the renderer config parser supports that alias.
- Keep `build_dataset_usd.num_workers: 1` and
  `max_concurrent_requests: 1` for one standalone OVRTX service.
- Put `chat_template_kwargs.enable_thinking=false` under both
  `identify_asset.vlm` and `predict.vlm` for Qwen3.5 35B NIM.
- Make the prediction prompt return top-level `physical_properties`, not
  `classification.physical_properties`. The current streaming prediction writer
  stores the full VLM response under the configured `output_key`, and
  `apply_physics` reads `classification.physical_properties`.

The skill does not ship a prebuilt H100 smoke config. Create one in ignored
local runtime storage, then edit it to apply the bullets above:

```bash
source .venv/bin/activate
SMOKE_DIR=.data/physics-agent-brev-qwen35
SMOKE_CONFIG="$SMOKE_DIR/lightbulb_brev_qwen35_h100.yaml"
mkdir -p "$SMOKE_DIR"
cp apps/physics_agent/configs/lightbulb.yaml "$SMOKE_CONFIG"
```

Run a clean first pass after editing `$SMOKE_CONFIG`:

```bash
source .venv/bin/activate
SMOKE_DIR=.data/physics-agent-brev-qwen35
SMOKE_CONFIG="$SMOKE_DIR/lightbulb_brev_qwen35_h100.yaml"
RENDER_ENDPOINT=http://localhost:8011 \
MA_RENDERING_USE_DATA_URI=true \
PA_VLM_NIM_BASE_URL=http://localhost:8013/v1 \
PA_LLM_NIM_BASE_URL=http://localhost:8013/v1 \
PA_VLM_MODEL=qwen/qwen3.5-35b-a3b \
PA_NIM_API_KEY=not-used \
WU_VLM_GENERATE_TIMEOUT_SECONDS=300 \
PA_IDENTIFY_ASSET_VLM_TIMEOUT=300 \
physics-agent run "$SMOKE_CONFIG" \
  --clean \
  --log-file "$SMOKE_DIR/run_qwen35_h100.log" \
  --log-level INFO
```

If the render/dataset steps already passed and only the prediction schema was
changed, avoid extra render spend by rerunning:

```bash
source .venv/bin/activate
SMOKE_DIR=.data/physics-agent-brev-qwen35
SMOKE_CONFIG="$SMOKE_DIR/lightbulb_brev_qwen35_h100.yaml"
RENDER_ENDPOINT=http://localhost:8011 \
MA_RENDERING_USE_DATA_URI=true \
PA_VLM_NIM_BASE_URL=http://localhost:8013/v1 \
PA_LLM_NIM_BASE_URL=http://localhost:8013/v1 \
PA_VLM_MODEL=qwen/qwen3.5-35b-a3b \
PA_NIM_API_KEY=not-used \
WU_VLM_GENERATE_TIMEOUT_SECONDS=300 \
PA_IDENTIFY_ASSET_VLM_TIMEOUT=300 \
physics-agent run "$SMOKE_CONFIG" \
  --skip identify_asset,build_dataset_usd \
  --log-file "$SMOKE_DIR/run_qwen35_h100_retry.log" \
  --log-level INFO
```

Expected smoke result for the lightbulb example: 8 dataset entries, 8
predictions with `physical_properties`, and an output USD with physics schemas
applied to all 8 prims. Keep `.data/physics-agent-brev-qwen35/` as local
runtime state and do not commit it.

## Output Format

Return the Brev instance names, render and VLM endpoint URLs, selected model
IDs, smoke-test results, Physics Agent environment exports, pipeline result,
and cleanup commands.

## Troubleshooting

- If OVRTX logs `VkResult: ERROR_INCOMPATIBLE_DRIVER`, the Brev image may be
  missing NVIDIA GL/Vulkan libraries. Install `libnvidia-gl-580` matching the
  driver, regenerate `/etc/cdi/nvidia.yaml`, copy it to `/var/run/cdi`, and
  recreate the render container.
- AWS `g7e.2xlarge` / RTX PRO Server 6000 and AWS `g6e.xlarge` / L40S are
  validated render candidates for OVRTX `/health` and `/render` smoke tests.
  Prefer RTX PRO Server 6000 for fast validation loops.
- Hyperstack H100 can show `health_status: UNHEALTHY` while
  `shell_status: READY` and the port-forwarded NIM endpoint works. Trust shell
  readiness plus endpoint smokes.
- Larger Qwen models are opt-in. Some larger Qwen 3.5 variants may have long
  startup time, high VRAM use, or leave stale GPU memory after failed startup.
- If Brev auth fails with `malformed refresh token` before deletion, power off
  reachable VMs over SSH to stop compute spend, then ask the user to re-login
  so `brev delete` can finish.

## Cleanup

Stop port-forwards and delete GPU nodes unless the user wants to keep them:

```bash
brev delete wu-pa-render
brev delete wu-pa-vlm-a100
brev delete wu-pa-vlm-h100
brev ls --json
```
