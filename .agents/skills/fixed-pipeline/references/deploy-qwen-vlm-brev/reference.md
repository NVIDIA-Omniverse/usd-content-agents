---
name: deploy-qwen-vlm-brev
description: "Fixed pipeline reference: Deploy or smoke-test a Brev-hosted OpenAI-compatible Qwen VLM/LLM endpoint. Use when the user asks to deploy VLM on Brev, host Qwen on A100/H100, create a Qwen VLM endpoint for Material or Physics Agent, test a Brev model node, or recreate the validated Denvr A100 or Hyperstack H100 Qwen deployment. This workflow uses a local port-forward so local agent pipelines can call the remote endpoint."
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - brev
  - qwen
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
compatibility: Requires Brev CLI access, Docker on a remote A100/H100-class GPU instance, Hugging Face or NGC credentials for the selected Qwen runtime, and local port-forward access.
---

# Fixed Pipeline Reference: Deploy Qwen VLM With Brev

## When to Use

Use this skill for a standalone Qwen VLM/LLM endpoint that local agent
pipelines can reach through `http://localhost:8003/v1`.

Use the pinned A100 vLLM path for low-cost endpoint smoke tests, Qwen3.5 35B
NIM on A100 or H100 for larger Material/Physics validation, and Qwen3.6 only
when the user asks for that experiment.

## Limitations

- Keep Hugging Face, NGC, and endpoint credentials in `.env` or remote env files
  with restrictive permissions; never print or commit them.
- Do not use `vllm/vllm-openai:latest`; it drifted to an incompatible
  CUDA/PyTorch build on the validated Denvr A100 driver 550 host.
- Disable Qwen thinking for direct Material and Physics Agent responses unless
  reasoning output is explicitly wanted.
- Qwen3.6 passed text and public image URL chat, but base64 data URL image
  prompts failed. Material Agent sends local rendered images through base64
  payloads, so Qwen3.6 is not a promoted Material Agent VLM.
- Do not promote another A100/H100 provider until it passes shell readiness,
  writable disk/cache qualification, `/v1/models`, text chat, image chat, and
  local port-forward smoke.

## Prerequisites

- Brev CLI access and an SSH-ready A100 or H100 GPU instance.
- Docker with enough writable storage for model images and Hugging Face or NIM
  cache directories.
- Hugging Face credentials for the pinned vLLM path, or NGC credentials for the
  NVIDIA NIM paths.
- Local port-forward access from the agent pipeline host to the remote endpoint.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Create or reuse the Brev node only after the dry-run looks acceptable.
3. Qualify storage, GPU, Docker root, and model cache before pulling images.
4. Start the selected Qwen runtime, then smoke test text and image chat.
5. Wire Material or Physics Agent environment variables and clean up the node.

## Credit Safety

Start with inventory and a dry-run:

```bash
brev ls --json
brev create wu-vlm-a100 --dry-run --type denvr_A100_sxm4_80G --min-disk 500
brev create wu-vlm-h100-hs --dry-run --type hyperstack_H100 --min-disk 500
```

Only after capacity and spend are acceptable:

```bash
brev create wu-vlm-a100 --type denvr_A100_sxm4_80G --min-disk 500 --timeout 1200
```

## Qualify Storage And GPU

Wait for `shell_status: READY`, then inspect the real writable filesystems
before pulling model images or weights:

```bash
brev exec wu-vlm-a100 \
  "df -h / /home /mnt/* 2>/dev/null || true; \
   lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; \
   du -sh /var/lib/docker ~/.cache/huggingface 2>/dev/null || true; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
```

On the validated Denvr image, root is small and `/dev/vdc` is the blank large
data disk. Only format it after confirming it is blank. Denvr hosts may not use
`ubuntu` as the SSH user, so own writable cache directories with the active
remote user:

```bash
brev exec wu-vlm-a100 "sudo file -s /dev/vdc && sudo fdisk -l /dev/vdc"
brev exec wu-vlm-a100 \
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

## Start vLLM

Use the pinned vLLM image on Denvr A100. The model is an
OpenAI-compatible VLM and also works for light LLM smoke tests.

```bash
brev exec wu-vlm-a100 \
  "docker rm -f qwen-vlm >/dev/null 2>&1 || true; \
   docker run -d --name qwen-vlm --gpus all --shm-size=32g -p 8000:8000 \
     -v /mnt/data/huggingface:/root/.cache/huggingface \
     vllm/vllm-openai:v0.8.5.post1 \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
     --host 0.0.0.0 --port 8000 --dtype bfloat16 \
     --max-model-len 8192 --gpu-memory-utilization 0.80 \
     --max-num-seqs 1 --trust-remote-code \
     --limit-mm-per-prompt image=20 --enforce-eager"
```

Poll readiness:

```bash
brev exec wu-vlm-a100 "docker logs --tail 120 qwen-vlm"
brev exec wu-vlm-a100 "curl -fsS http://localhost:8000/v1/models"
```

## Local Port-Forward And Smoke

Keep this running while local pipelines call the endpoint:

```bash
brev port-forward wu-vlm-a100 -p 8003:8000
```

From another local shell:

```bash
curl -fsS http://localhost:8003/v1/models
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-VL-7B-Instruct","messages":[{"role":"user","content":"Reply with exactly: brev-ok"}],"max_tokens":16,"temperature":0}'
SMOKE_IMAGE_DATA="$(python .agents/skills/fixed-pipeline/scripts/generate_vlm_smoke_image.py)"
jq -nc --arg image_url "$SMOKE_IMAGE_DATA" \
  '{model:"Qwen/Qwen2.5-VL-7B-Instruct",messages:[{role:"user",content:[
    {type:"text",text:"What is the dominant color in this image? Answer with one word."},
    {type:"image_url",image_url:{url:$image_url}}
  ]}],max_tokens:16,temperature:0}' |
  curl -fsS http://localhost:8003/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

The image smoke should return a short color answer such as `red`.

## Start NVIDIA NIM 35B

Use this path when the user explicitly wants the larger NVIDIA NIM VLM. It
requires an `NGC_API_KEY`; do not print or commit the key. Export it from an
approved secret source before running the auth block; never source the repo
`.env` as shell code.

On this Denvr image, Docker does not define the `nvidia` runtime name even
though `--gpus all` works. Do not use `--runtime=nvidia` here.

Before sending credentials, verify the node's SSH host-key fingerprint through
an independently authenticated Brev/provider channel and install that exact
key in `~/.ssh/known_hosts`. Do not use `ssh-keyscan` as the trust source.

```bash
(
  set -e
  [ -n "${NGC_API_KEY:-}" ] || { echo "NGC_API_KEY is required" >&2; exit 1; }
  cleaned="$(printf %s "$NGC_API_KEY" | tr -d "\r\n")"; [ "$cleaned" = "$NGC_API_KEY" ] || { echo "NGC_API_KEY must be a single-line value" >&2; exit 1; }
  ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$HOME/.ssh/known_hosts")
  ssh "${ssh_options[@]}" wu-vlm-a100 true || { echo "A trusted SSH host key for wu-vlm-a100 is required before sending credentials" >&2; exit 1; }
  printf 'NGC_API_KEY=%s\n' "$NGC_API_KEY" | \
    ssh "${ssh_options[@]}" wu-vlm-a100 'set -e; umask 077; destination="$HOME/.ngc-nim.env"; tmp_env="$(mktemp "${destination}.XXXXXX")"; cleanup() { rm -f -- "$tmp_env"; }; trap cleanup EXIT; trap "exit 1" HUP INT TERM; cat > "$tmp_env"; chmod 600 "$tmp_env"; mv -f -- "$tmp_env" "$destination"; trap - EXIT HUP INT TERM'
  printf '%s\n' "$NGC_API_KEY" | \
    ssh "${ssh_options[@]}" wu-vlm-a100 'docker login nvcr.io -u \$oauthtoken --password-stdin'
)
```

Create a writable NIM cache on the mounted data disk:

```bash
ssh wu-vlm-a100 \
  'sudo mkdir -p /mnt/data/nim && \
   sudo chown -R $(id -un):$(id -gn) /mnt/data/nim'
```

Start the container:

```bash
ssh wu-vlm-a100 \
  'docker rm -f qwen35b-nim >/dev/null 2>&1 || true; \
   docker run -d --name qwen35b-nim --gpus all --shm-size=32GB \
     --env-file ~/.ngc-nim.env \
     -e NIM_SERVER_PORT=8000 \
     -e NIM_MAX_IMAGES_PER_PROMPT=20 \
     -e HOME=/tmp \
     -e USER=$(id -un) \
     -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
     -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
     -e PYTHONMULTIPROCESSING_START_METHOD=spawn \
     -u $(id -u) \
     -v /mnt/data/nim:/opt/nim/.cache \
     -p 8000:8000 \
     nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant'
```

Poll readiness:

```bash
ssh wu-vlm-a100 \
  'docker logs --tail 160 qwen35b-nim'
ssh wu-vlm-a100 \
  'curl -fsS http://localhost:8000/v1/models'
```

`NIM_MAX_IMAGES_PER_PROMPT=20` overrides NIM defaults that can be as low as five images.

Keep a local port-forward open:

```bash
brev port-forward wu-vlm-a100 -p 8003:8000
```

Smoke the endpoint with thinking disabled per request:

```bash
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3.5-35b-a3b","messages":[{"role":"user","content":"Reply with exactly: brev-ok"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3.5-35b-a3b","messages":[{"role":"user","content":[{"type":"text","text":"What is the dominant color in this image? Answer with one word."},{"type":"image_url","image_url":{"url":"data:image/gif;base64,R0lGODlhAQABAPAAAP8AAAAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="}}]}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

## Start NVIDIA NIM 35B On H100

Use this path when the user asks for the H100 NIM golden path. Start with
inventory and dry-run:

```bash
brev ls --json
brev create wu-vlm-h100-hs --dry-run --type hyperstack_H100 --min-disk 500
```

After confirming spend:

```bash
brev create wu-vlm-h100-hs --type hyperstack_H100 --min-disk 500 --timeout 1200
```

Qualify the host. The validated Hyperstack image uses `/ephemeral` for the
large writable disk and may report `health_status: UNHEALTHY` even when
`shell_status: READY`; trust shell plus endpoint smokes.

```bash
brev exec wu-vlm-h100-hs \
  "df -h / /ephemeral; \
   lsblk -b -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
```

Move Docker, containerd, and NIM cache to `/ephemeral`. On the validated
Hyperstack image, setting Docker data-root alone was not enough because image
layers were stored through containerd under `/var/lib/containerd`.

```bash
brev exec wu-vlm-h100-hs \
  "set -e; \
   sudo systemctl stop docker || true; \
   sudo systemctl stop containerd; \
   sudo mkdir -p /ephemeral/docker /ephemeral/nim /ephemeral/nim-workspace; \
   sudo chown -R \$(id -un):\$(id -gn) /ephemeral/nim /ephemeral/nim-workspace; \
   if [ -d /var/lib/containerd ] && [ ! -e /ephemeral/containerd ]; then \
     sudo mv /var/lib/containerd /ephemeral/containerd; \
   elif [ -d /var/lib/containerd ]; then \
     sudo rsync -aHAX --delete /var/lib/containerd/ /ephemeral/containerd/ && \
     sudo rm -rf /var/lib/containerd; \
   fi; \
   printf 'disabled_plugins = [\"cri\"]\nroot = \"/ephemeral/containerd\"\nstate = \"/run/containerd\"\n' | \
     sudo tee /etc/containerd/config.toml >/dev/null; \
   sudo env DOCKER_DATA_ROOT=/ephemeral/docker python3 - <<'PY'
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
   sudo systemctl start containerd && \
   sudo systemctl start docker && \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}' && \
   df -h / /ephemeral"
```

Copy the NGC key and log in without printing the token. Export it from an
approved secret source first; never source the repo `.env` as shell code.
Verify the node's SSH host-key fingerprint through an independently
authenticated Brev/provider channel and install that exact key in
`~/.ssh/known_hosts`; do not use `ssh-keyscan` as the trust source.

```bash
(
  set -e
  [ -n "${NGC_API_KEY:-}" ] || { echo "NGC_API_KEY is required" >&2; exit 1; }
  cleaned="$(printf %s "$NGC_API_KEY" | tr -d "\r\n")"; [ "$cleaned" = "$NGC_API_KEY" ] || { echo "NGC_API_KEY must be a single-line value" >&2; exit 1; }
  ssh_options=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$HOME/.ssh/known_hosts")
  ssh "${ssh_options[@]}" wu-vlm-h100-hs true || { echo "A trusted SSH host key for wu-vlm-h100-hs is required before sending credentials" >&2; exit 1; }
  printf 'NGC_API_KEY=%s\n' "$NGC_API_KEY" | \
    ssh "${ssh_options[@]}" wu-vlm-h100-hs 'set -e; umask 077; destination="$HOME/.ngc-nim.env"; tmp_env="$(mktemp "${destination}.XXXXXX")"; cleanup() { rm -f -- "$tmp_env"; }; trap cleanup EXIT; trap "exit 1" HUP INT TERM; cat > "$tmp_env"; chmod 600 "$tmp_env"; mv -f -- "$tmp_env" "$destination"; trap - EXIT HUP INT TERM'
  printf '%s\n' "$NGC_API_KEY" | \
    ssh "${ssh_options[@]}" wu-vlm-h100-hs 'docker login nvcr.io -u \$oauthtoken --password-stdin'
)
```

Start the NIM. Keep `VLLM_ENABLE_CUDA_COMPATIBILITY=1`; it is required on the
validated Hyperstack image. Set `NIM_MAX_MODEL_LEN=65536` for agent smokes;
the default 262,144-token context can be killed during H100 startup. Set
`NIM_MAX_IMAGES_PER_PROMPT=20` for agent multi-view prompts.

```bash
ssh wu-vlm-h100-hs \
  'sudo mkdir -p /ephemeral/nim /ephemeral/nim-workspace && \
   sudo chown -R $(id -un):$(id -gn) /ephemeral/nim /ephemeral/nim-workspace; \
   docker rm -f qwen35b-nim >/dev/null 2>&1 || true; \
   docker run -d --name qwen35b-nim --gpus all --shm-size=32GB \
     --env-file ~/.ngc-nim.env \
     -e NIM_SERVER_PORT=8000 \
     -e NIM_MAX_MODEL_LEN=65536 \
     -e NIM_MAX_IMAGES_PER_PROMPT=20 \
     -e HOME=/tmp \
     -e USER=$(id -un) \
     -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
     -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
     -e PYTHONMULTIPROCESSING_START_METHOD=spawn \
     -u $(id -u) \
     -v /ephemeral/nim:/opt/nim/.cache \
     -v /ephemeral/nim-workspace:/opt/nim/workspace \
     -p 8000:8000 \
     nvcr.io/nim/qwen/qwen3.5-35b-a3b:1.7.0-variant'
```

Poll readiness. First cold start may spend several minutes in DeepGEMM warmup
and CUDA graph capture after model weights load. If `/v1/health/live` resets
the connection while `docker exec qwen35b-nim ps` shows `ptxas` or DeepGEMM
warmup, keep waiting:

```bash
ssh wu-vlm-h100-hs \
  'docker logs --tail 160 qwen35b-nim'
ssh wu-vlm-h100-hs \
  'curl -fsS http://localhost:8000/v1/health/ready'
ssh wu-vlm-h100-hs \
  'curl -fsS http://localhost:8000/v1/models'
```

Observed H100 startup with `NIM_MAX_MODEL_LEN=65536`: FP8, `tp=1`, about 31.45 GiB KV cache and 22.62x full-context concurrency.

Run text and image smokes locally or remotely with
`chat_template_kwargs.enable_thinking=false`, then port-forward. Run the
20-image smoke before an agent test that uses multi-view prompts:

```bash
brev port-forward wu-vlm-h100-hs -p 8003:8000
curl -fsS http://localhost:8003/v1/models
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3.5-35b-a3b","messages":[{"role":"user","content":"Reply with exactly: brev-ok"}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
curl -fsS http://localhost:8003/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen/qwen3.5-35b-a3b","messages":[{"role":"user","content":[{"type":"text","text":"What is the dominant color in this image? Answer with one word."},{"type":"image_url","image_url":{"url":"data:image/gif;base64,R0lGODlhAQABAPAAAP8AAAAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw=="}}]}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

If the next workload depends on many rendered views in one prompt, verify the
container env and run the 20-image smoke in `references/qwen35-h100-smoke.md`
before handing the endpoint to the agent.

## Experimental Qwen3.6 35B On H100

Use this path only when the user explicitly wants to test Qwen3.6. Read
`references/qwen36-h100.md` for the container command and smoke-test details.
Public image URLs can pass while base64 data URLs fail; for Material Agent that
failure is a blocker, not a warning.

## Agent Wiring

For the pinned vLLM path, use the existing small-model wiring.

For Material Agent:

```bash
export MA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export MA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export MA_NIM_API_KEY=not-used
```

For Physics Agent:

```bash
export PA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export PA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_NIM_API_KEY=not-used
```

For the 35B NIM path, use the NIM model ID and make sure requests include
`chat_template_kwargs.enable_thinking=false`. The endpoint environment
variables route traffic and select the model; they do not by themselves inject
request-level `chat_template_kwargs`.

Material Agent:

```bash
export MA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_VLM_MODEL=qwen/qwen3.5-35b-a3b
export MA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export MA_LLM_MODEL=qwen/qwen3.5-35b-a3b
export MA_NIM_API_KEY=not-used
```

Physics Agent:

```bash
export PA_VLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_VLM_MODEL=qwen/qwen3.5-35b-a3b
export PA_LLM_NIM_BASE_URL=http://localhost:8003/v1
export PA_NIM_API_KEY=not-used
```

Use `MA_NIM_API_KEY=not-used` and `PA_NIM_API_KEY=not-used` only for local
no-auth endpoints such as Brev port-forwards. For tunnel, external URL,
private-IP, or otherwise authenticated endpoints, put the real key in `.env`
instead of the dummy value.

Patch the agent model config, or pass the equivalent extra request body, so
Qwen thinking is disabled for direct agent responses:

```yaml
vlm:
  chat_template_kwargs:
    enable_thinking: false
llm:
  chat_template_kwargs:
    enable_thinking: false
```

## Output Format

Return the Brev instance name, GPU type, endpoint URL, selected model ID, text
and image smoke status, agent exports, and cleanup command.

## Troubleshooting
- If startup fails, report the failing command, recent container log tail, and
  storage/GPU qualification result.
- If image chat fails, include the HTTP status, response summary, model ID, and
  whether data URL transport is the likely blocker.

## Cleanup

Stop port-forward sessions and delete the VM unless the user explicitly wants
to keep it:

```bash
brev delete wu-vlm-a100
brev delete wu-vlm-h100-hs
brev ls --json
```
