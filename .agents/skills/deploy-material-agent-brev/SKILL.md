---
name: deploy-material-agent-brev
description: Deploy or smoke-test the Material Agent with Brev-hosted dependency endpoints. Use when the user asks to test material agent on Brev, deploy material agent with Brev, use RTX/L40S for rendering and A100/H100-grade GPU for VLM, run a Brev hybrid material-agent test, or recreate the Brev material-agent deployment. This workflow keeps the main pipeline local and uses Brev port-forwards to reach OVRTX and Qwen-family VLM endpoints.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - material-agent
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
compatibility: Requires Brev CLI access, a remote RTX/L40S-class render GPU for OVRTX, an A100/H100-class VLM endpoint, local port-forwards, and Material Agent provider credentials.
---

# Deploy Material Agent With Brev

## When to Use

Use this workflow for the credit-conscious hybrid path:

- Local machine runs the material-agent pipeline.
- Brev `wu-ma-render` runs standalone OVRTX on an RTX GPU. Prefer AWS
  `g7e.2xlarge` / RTX PRO Server 6000 for fast validation, or AWS
  `g6e.xlarge` / L40S when lower cost matters more than cold-start latency.
- Brev `wu-ma-vlm-a100` or `wu-ma-vlm-h100` runs an OpenAI-compatible Qwen
  VLM/LLM endpoint. The validated larger path is H100 + Qwen3.5 35B NIM with
  `NIM_MAX_MODEL_LEN=65536`; it passed Material Agent end-to-end with local
  data-URI rendered image payloads.
- Local port-forwards expose `RENDER_ENDPOINT=http://localhost:8001` and
  `MA_VLM_NIM_BASE_URL=http://localhost:8003/v1`.
- Optional image-generation endpoints exposed through `MA_IMAGE_GEN_*` are used
  by generated reference images and generated material-library textures. Reuse a
  Texture Agent image-generation endpoint when one is already running.

## Limitations

- Keep provider credentials in `.env` or remote env files with restrictive
  permissions; never print or commit them.
- Prefer local port-forwards for this hybrid path; do not assume Brev
  instance-to-instance networking is available.
- Delete GPU nodes after validation unless the user asks to keep them.

## Prerequisites

- Brev CLI access and SSH-ready render and VLM nodes.
- Docker, NVIDIA Container Toolkit, and writable storage on each remote node.
- Local Material Agent environment with the provider keys and optional Scene
  Optimizer package required by the selected config.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Create or reuse the Brev render node, then validate OVRTX health and render.
3. Create or reuse the VLM node, then validate model listing and image chat.
4. Set local Material Agent render, VLM, LLM, and optional image-generation environment variables.
5. Run the local pipeline with the forwarded endpoints, then clean up nodes and
   port-forwards.

## Credit Safety

Always start with:

```bash
brev --version
brev healthcheck
brev ls --json
python scripts/brev_agent_services.py --service material --preset hybrid --render-gpu-name RTX --vlm-gpu-name A100
```

If instances already exist, inspect and reuse them instead of creating new
ones. Use `brev create --dry-run` before real creation unless the user has
already confirmed the exact spend.

For service-on-Brev presets, the planner excludes local `.env` files during
worktree copy, then writes a minimal remote `.env` with generated endpoint/model
wiring and starts Docker Compose with `--env-file .env`. Edit that remote
`.env` before the Compose step when a generated comment asks for a real API key.

Useful dry-run candidates:

```bash
brev create wu-ma-render --dry-run --type g7e.2xlarge --min-disk 500
brev create wu-ma-render --dry-run --type g6e.xlarge --min-disk 500
brev create wu-ma-render --dry-run --type gpu-l40s-a.1gpu-8vcpu-32gb --min-disk 500
brev create wu-ma-vlm-a100 --dry-run --type denvr_A100_sxm4_80G --min-disk 500
brev create wu-ma-vlm-h100 --dry-run --type hyperstack_H100 --min-disk 500
```

Only after confirmation, create the render node plus one model node. Do not
create both A100 and H100 unless the test explicitly needs both:

```bash
brev create wu-ma-render --type g7e.2xlarge --min-disk 500 --timeout 1200
# Small/cheap VLM smoke option:
brev create wu-ma-vlm-a100 --type denvr_A100_sxm4_80G --min-disk 500 --timeout 1200
# Larger validated Material Agent VLM option:
brev create wu-ma-vlm-h100 --type hyperstack_H100 --min-disk 500 --timeout 1200
```

## OVRTX Render Node

Wait until `brev ls --json` shows shell access ready, then copy the repo to a
writable path. Brev's default user is `ubuntu`; `/workspace` was not writable
on observed images, so use `~/world-understanding`.

```bash
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .data \
  --exclude .build-resources \
  --exclude docs/metrics --exclude coverage.xml \
  --exclude .env --exclude '.env.*' --exclude '.env-*' \
  --exclude '.env *' --exclude .envrc \
  --exclude '*credentials*.json' --exclude '*private*key*' \
  --exclude '*.key' --exclude '*.pem' --exclude '*.p12' --exclude '*.p8' \
  --exclude id_rsa --exclude id_dsa --exclude id_ecdsa --exclude id_ed25519 \
  ./ wu-ma-render:~/world-understanding/
brev exec wu-ma-render "mkdir -p ~/world-understanding/.build-resources"
brev exec wu-ma-render "cd ~/world-understanding && OVRTX_RENDER_MODE=pt OVRTX_NUM_SENSOR_UPDATES=1 OVRTX_DAEMON_RENDER_TIMEOUT=900 docker compose -f apps/ovrtx_rendering_api/docker-compose.yml up -d --build"
brev port-forward wu-ma-render -p 8001:8001
curl -fsS http://localhost:8001/health
```

Keep the port-forward running while the local pipeline runs. OVRTX can take
several minutes to warm up before `gpu_initialized` is true.
The empty `.build-resources` directory is required because the OVRTX Dockerfile
copies that path during build; keep the rsync exclusion so local/internal Scene
Optimizer bundles and other build artifacts are not copied to the render node.
Do not use `rt2` for Material Agent validation. For a first render-service
return-path smoke, it is acceptable to add `OVRTX_NUM_SENSOR_UPDATES=1` while
keeping `OVRTX_RENDER_MODE=pt`; restore the default before quality-sensitive
pipeline runs. Avoid very short cold-start render timeouts. A 900 second
timeout allowed warm-up and an OVRTX `/render` smoke to pass on validated AWS
RTX PRO Server 6000 and L40S nodes, but L40S can be noticeably slower.

## A100/H100 VLM Node

Start a Qwen-family OpenAI-compatible VLM/LLM endpoint on the model node. The
validated path is Denvr `denvr_A100_sxm4_80G` with `Qwen/Qwen2.5-VL-7B-Instruct`. Promote
H100 providers or larger Qwen 3.5/3.6 models only after the selected image
passes disk qualification, endpoint readiness, text chat, image chat, and a
Material Agent predict smoke.
If the user asks for NVIDIA NIM, use `deploy-qwen-vlm-brev`. The validated
larger Material Agent path is `hyperstack_H100` with
`nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant`, model ID
`qwen/qwen3.5-35b-a3b`, `NIM_MAX_MODEL_LEN=65536`,
`NIM_MAX_IMAGES_PER_PROMPT=20`, `/ephemeral` for Docker/NIM cache, and
`VLLM_ENABLE_CUDA_COMPATIBILITY=1`. That NIM requires request-level
`chat_template_kwargs: {"enable_thinking": false}` for direct answers.

The `nvcr.io/nim/qwen/qwen3.6-35b-a3b:1.7.0-variant` H100 path can serve text
and public image URLs, but it rejects Material Agent's local/base64 image
payloads with HTTP 400 (`Input should be a valid string`). Do not use Qwen3.6
35B A3B as the Material Agent default until the image transport is changed or a
compatible container is found. Prefer the validated Qwen3.5 35B NIM endpoint
for Material Agent VLM tests that need the larger model.

Qualify the provider image before pulling model weights. Require a real
writable Docker/Hugging Face cache location with enough free disk; use at least
250 GB free for first small-Qwen smoke tests and more for 9B+ experiments.

```bash
brev exec wu-ma-vlm-a100 \
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

On the validated Denvr A100 image, `/dev/vdc` is the blank large data disk.
Only format it after confirming it is blank:

```bash
brev exec wu-ma-vlm-a100 \
  "sudo file -s /dev/vdc && sudo fdisk -l /dev/vdc"
brev exec wu-ma-vlm-a100 \
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
```

This merge step preserves existing Docker daemon keys such as `runtimes` and
`default-runtime`; `nvidia-ctk` runs before Docker is restarted so the daemon
comes back with the NVIDIA runtime still registered after the data-root move.
Do not replace
`/etc/docker/daemon.json` with a one-key file on provider images.

Serve the pinned Qwen vLLM endpoint with multimodal prompts enabled:

```bash
brev exec wu-ma-vlm-a100 \
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
brev exec wu-ma-vlm-a100 "docker logs --tail 100 qwen-vlm"
```

Required local wiring:

```bash
brev exec wu-ma-vlm-a100 "curl -fsS http://localhost:8000/v1/models"
brev port-forward wu-ma-vlm-a100 -p 8003:8000
curl -fsS http://localhost:8003/v1/models
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":[{"type":"text","text":"What is the dominant color in this image? Answer with one word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="}}]}],"max_tokens":16,"temperature":0}'
```

Keep the port-forward running while the local pipeline runs.

## Local Material Pipeline Environment

Set these locally before running the material-agent CLI or service client:

```bash
export RENDER_ENDPOINT=http://localhost:8001
export MA_RENDERING_USE_DATA_URI=true
export MA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export MA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export MA_NIM_API_KEY=not-used
```

`MA_RENDERING_USE_DATA_URI=true` is required when the local Material Agent
pipeline uses the NVCF-compatible `remote` render backend against a
port-forwarded OVRTX service. It embeds prepared USD stages in the render
request and avoids S3/NVCF staging.
Use `MA_NIM_API_KEY=not-used` only for local no-auth VLM/LLM endpoints such as
a Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `MA_NIM_API_KEY` in `.env` instead of the
dummy key.

If the local Scene Optimizer package is not installed, either fetch it before
the run or skip optimization for small smoke assets:

```bash
material-agent run <config.yaml> --skip optimize_usd
```

For the validated 35B NIM path, set `MA_VLM_MODEL` and `MA_LLM_MODEL` to
`qwen/qwen3.5-35b-a3b` and ensure the model config or request extra body passes
`chat_template_kwargs.enable_thinking=false`. A ladder end-to-end smoke passed
with RTX PRO Server 6000 OVRTX plus H100 Qwen3.5 35B NIM: 20 dataset renders,
4 VLM predictions, material application, output validation, and 2 final
renders.

This path does not require Brev-to-Brev networking because the local machine is
the caller for both remote endpoints.

## Output Format

Return the Brev instance names, render and VLM endpoint URLs, selected model
IDs, smoke-test results, Material Agent environment exports, pipeline result,
and cleanup commands.

## Troubleshooting

- Use local port-forwards for this hybrid path; do not rely on Brev private
  instance-to-instance networking unless it has been proven for the selected
  provider.
- Wait for `shell_status: READY` before `brev exec` or `rsync`.
- Use `~/world-understanding` and rsync with secret/cache excludes; `/workspace`
  may not be writable.
- If a render provider image lacks NVIDIA GL/Vulkan libraries, install the
  matching `libnvidia-gl` package, add `vulkan-tools`, regenerate CDI, and
  recreate the OVRTX container.
- Start with `Qwen/Qwen2.5-VL-7B-Instruct` for self-hosted endpoint smoke tests. Move to
  `Qwen/Qwen3.5-9B` or larger models only after the endpoint is proven on the
  selected Brev image.
- Check usable writable disk before downloading Hugging Face models; requested
  Brev disk size may not be reflected in the root filesystem.
- If Brev auth fails before deletion, power off reachable VMs over SSH
  (`ssh <name> "sudo poweroff"`) and re-login so `brev delete` can finish.

## Cleanup

When the test is complete, stop port-forward sessions and delete the GPU nodes
unless the user explicitly wants to keep them:

```bash
brev delete wu-ma-render
brev delete wu-ma-vlm-a100
brev delete wu-ma-vlm-h100
brev ls --json
```
