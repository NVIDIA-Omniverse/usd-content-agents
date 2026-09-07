---
name: deploy-texture-agent-brev
description: "Fixed pipeline reference: Deploy or smoke-test the Texture Agent with Brev-hosted endpoints. Use when the user asks to test texture agent on Brev, deploy texture agent with Brev, run a Brev texture-agent service-only or hybrid test, host a small Qwen-family LLM for Texture Agent, or recreate the Brev texture-agent deployment. Texture Agent does not need OVRTX rendering by default."
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - texture-agent
  - brev
  - image-generation
  - deployment
tools:
  - Shell
  - Docker
  - Python
  - curl
  - Filesystem
  - brev
  - ssh
compatibility: Requires Brev CLI access, remote image-generation and optional LLM endpoints, Texture Agent provider credentials, Docker for service-only validation, and local port-forward access.
---

# Fixed Pipeline Reference: Deploy Texture Agent With Brev

## When to Use

Texture Agent does not require OVRTX rendering. The target path for this
worktree is a local Texture Agent pipeline with Brev-hosted dependency
endpoints:

- Brev image-generation endpoint: required for `generate_textures`.
- Brev small Qwen-family LLM endpoint: optional, only needed when the config
  does not provide explicit prompts for every material.
- Service-on-Brev modes remain secondary validation paths, not the default.

## Limitations

- Keep provider credentials in `.env` or remote env files with restrictive
  permissions; never print or commit them.
- Prefer the hybrid local pipeline path; service-on-Brev is a secondary
  validation mode.
- Delete GPU nodes after validation unless the user asks to keep them.

## Prerequisites

- Brev CLI access and SSH-ready image-generation and optional LLM nodes.
- Docker and writable storage on remote nodes used for service-only or sidecar
  validation.
- Local Texture Agent environment with image-generation and optional LLM
  provider credentials.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Inspect Brev state and choose the hybrid path unless service-on-Brev is
   explicitly requested.
3. Start or reuse image-generation and optional LLM endpoints.
4. Run the local Texture Agent smoke or the secondary service-only validation.
5. Report endpoint wiring and clean up nodes and port-forwards.

## Credit Safety

Start with:

```bash
brev --version
brev healthcheck
brev ls --json
python scripts/brev_agent_services.py --service texture --preset service-only
python scripts/brev_agent_services.py --service texture --preset hybrid
```

Reuse existing instances when possible. Run `brev create --dry-run` before
real creation unless the user already confirmed spend.

For service-on-Brev presets, the planner excludes local `.env` files during
worktree copy, then writes a minimal remote `.env` with generated endpoint/model
wiring and starts Docker Compose with `--env-file .env`. Edit that remote
`.env` before the Compose step when a generated comment asks for a real API key.

The `single-host-local-sidecars` preset runs local FLUX image-gen and Nemotron
Nano LLM sidecars, not OVRTX rendering. It may use A100/H100-class
model-serving GPUs when a suitable multi-GPU Brev type is available; the RTX
requirement applies only to presets that run OVRTX.

## Hybrid Local Pipeline Path

Use the validated AWS `g6e.xlarge` L40S node for the FLUX image-generation
endpoint. A generic cheapest `L40S` search selected a Nebius image that failed
before shell readiness during validation, so prefer the exact AWS type until
another cheaper provider is proven.

Copy only `NGC_API_KEY` and `HF_TOKEN` to the remote NIM env file; do not copy
the full local `.env`. FLUX NIM needs `NGC_API_KEY` for the container pull and
`HF_TOKEN` for model weight access. Export both values from an approved secret
source before running this block. If they exist only in the repo `.env`, use
the repo planner, which parses only these keys without sourcing the file:

```bash
source .venv/bin/activate
python scripts/brev_agent_services.py --service texture --preset hybrid --name wu-ta \
  --image-gen-node-name wu-ta-image-gen
```

Before sending credentials, verify the node's SSH host-key fingerprint through
an independently authenticated Brev/provider channel and install that exact
key in `~/.ssh/known_hosts`.

```bash
brev create wu-ta-image-gen --dry-run --type g6e.xlarge --min-disk 500
brev create wu-ta-image-gen --type g6e.xlarge --min-disk 500 --timeout 1200
brev exec wu-ta-image-gen \
  "df -h / /home /mnt/* 2>/dev/null || true; \
   df -h /opt/dlami/nvme 2>/dev/null || true; \
   lsblk -f; \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
brev exec wu-ta-image-gen \
  "set -e; \
   sudo mkdir -p /opt/dlami/nvme/docker /opt/dlami/nvme/nim-cache && \
   sudo chown -R \$(id -un):\$(id -gn) /opt/dlami/nvme/nim-cache && \
   sudo env DOCKER_DATA_ROOT=/opt/dlami/nvme/docker python3 - <<'PY'
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
   sudo chmod -R u+rwX,g+rwX,o-rwx /opt/dlami/nvme/nim-cache && \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}' && \
   df -h / /opt/dlami/nvme"
(
  set -e -o pipefail
  if [ -z "${NGC_API_KEY:-}" ]; then
    echo "NGC_API_KEY is required" >&2
    exit 1
  fi
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is required" >&2
    exit 1
  fi
  validate_credential() {
    if [ -z "$2" ]; then
      echo "$1 is required" >&2
      return 1
    fi
    cleaned="$(printf %s "$2" | tr -d "\r\n")"
    if [ "$cleaned" != "$2" ]; then
      echo "$1 must be a single-line value" >&2
      return 1
    fi
  }
  validate_credential NGC_API_KEY "$NGC_API_KEY"
  validate_credential HF_TOKEN "$HF_TOKEN"
  ssh_options=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=$HOME/.ssh/known_hosts"
  )
  if ! ssh "${ssh_options[@]}" wu-ta-image-gen true; then
    echo "A trusted SSH host key for wu-ta-image-gen is required before sending credentials" >&2
    exit 1
  fi
  printf "NGC_API_KEY=%s\nHF_TOKEN=%s\n" "$NGC_API_KEY" "$HF_TOKEN" | \
    ssh "${ssh_options[@]}" wu-ta-image-gen '
      set -e
      umask 077
      destination="$HOME/.ngc-nim.env"
      tmp_env="$(mktemp "${destination}.XXXXXX")"
      cleanup() { rm -f -- "$tmp_env"; }
      trap cleanup EXIT
      trap "exit 1" HUP INT TERM
      cat > "$tmp_env"
      chmod 600 "$tmp_env"
      mv -f -- "$tmp_env" "$destination"
      trap - EXIT HUP INT TERM
    '
  ssh "${ssh_options[@]}" wu-ta-image-gen \
  "chmod 600 \"\$HOME/.ngc-nim.env\" && \
   NGC_API_KEY=\"\$(sed -n 's/^NGC_API_KEY=//p' \"\$HOME/.ngc-nim.env\")\" && \
   test -n \"\$NGC_API_KEY\" && \
   printf '%s\n' \"\$NGC_API_KEY\" | \
     docker login nvcr.io -u '\$oauthtoken' --password-stdin"
  ssh "${ssh_options[@]}" wu-ta-image-gen \
  "docker rm -f flux-image-gen >/dev/null 2>&1 || true; \
   docker run -d --name flux-image-gen --gpus all --shm-size=32g -p 8000:8000 \
     --env-file \"\$HOME/.ngc-nim.env\" \
     -e NIM_CACHE_PATH=/opt/nim/.cache \
     -v /opt/dlami/nvme/nim-cache:/opt/nim/.cache \
     nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant"
  brev exec wu-ta-image-gen \
    'for attempt in $(seq 1 60); do curl -fsS http://localhost:8000/v1/health/ready && exit 0; sleep 20; done; exit 1'
  brev port-forward wu-ta-image-gen -p 8005:8000
)
```

Use an L4-class node for the small Qwen-family LLM when auto-prompts are
needed. Skip this node when `material_textures` provides explicit prompts for
every material:

```bash
brev create wu-ta-llm --dry-run --gpu-name L4 --min-vram 24 --min-disk 500 --sort price --stoppable
brev create wu-ta-llm --gpu-name L4 --min-vram 24 --min-disk 500 --sort price --stoppable --timeout 1200
ssh wu-ta-llm \
  "df -h / /home /mnt/* 2>/dev/null || true; \
   lsblk -f; \
   docker info --format 'Docker Root Dir: {{.DockerRootDir}}'; \
   du -sh /var/lib/docker ~/.cache/huggingface 2>/dev/null || true; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
ssh wu-ta-llm \
  "docker rm -f qwen-llm >/dev/null 2>&1 || true; \
   docker run -d --name qwen-llm --gpus all --shm-size=32g -p 8000:8000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     vllm/vllm-openai:v0.8.5.post1 \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
     --host 0.0.0.0 --port 8000 --dtype bfloat16 \
     --max-model-len 8192 --gpu-memory-utilization 0.80 \
     --max-num-seqs 1 --trust-remote-code \
     --limit-mm-per-prompt image=20 --enforce-eager"
ssh wu-ta-llm "curl -fsS http://localhost:8000/v1/models"
brev port-forward wu-ta-llm -p 8003:8000
```

Set locally:

```bash
export TA_IMAGE_GEN_BACKEND=openai
export TA_IMAGE_GEN_BASE_URL=http://localhost:8005/v1
export TA_IMAGE_GEN_MODEL=black-forest-labs/flux.2-klein-4b
export TA_IMAGE_GEN_API_KEY=not-used
```

Use `TA_IMAGE_GEN_API_KEY=not-used` only for local no-auth endpoints such as a
planner-created Brev port-forward. When `--image-gen-base-url` points at an
existing HTTPS, tunnel, or otherwise authenticated endpoint, put the real
`TA_IMAGE_GEN_API_KEY` in `.env` and do not use the dummy key.

The Brev planner skips the optional LLM node by default. When auto-prompt
generation is required, rerun it with `--texture-include-llm`, start the small
Qwen endpoint, and add:

```bash
export TA_LLM_BACKEND=nim
export TA_LLM_BASE_URL=http://localhost:8003/v1
export TA_LLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export TA_NIM_API_KEY=not-used
```

Use `TA_NIM_API_KEY=not-used` only for local no-auth LLM endpoints such as a
Brev port-forward. For tunnel, external URL, private-IP, or otherwise
authenticated endpoints, put the real `TA_NIM_API_KEY` or `TA_LLM_API_KEY` in
`.env` instead of the dummy key.

When the config supplies explicit prompts for every material, keep the default
planner output without `TA_LLM_*`. For a public/staging-safe smoke, use the
shipped texture example config; it provides explicit prompts and can run through
the FLUX image-gen endpoint without the optional LLM node:

```bash
source .venv/bin/activate
PYTHONPATH=apps/texture_agent:. \
  TA_IMAGE_GEN_BACKEND=openai \
  TA_IMAGE_GEN_BASE_URL=http://localhost:8005/v1 \
  TA_IMAGE_GEN_MODEL=black-forest-labs/flux.2-klein-4b \
  TA_IMAGE_GEN_API_KEY=not-used \
  python -m texture_agent run apps/texture_agent/configs/texture_example.yaml \
    --session-id texture_example_brev_g6e -v
```

## Service-Only Path

Use this only when the service itself should run on Brev for secondary
validation:

```bash
brev create wu-ta-service --dry-run --type n2d-standard-4
brev create wu-ta-service --type n2d-standard-4
rsync -az --delete \
  --exclude .git --exclude .venv --exclude .data \
  --exclude docs/metrics --exclude coverage.xml \
  --exclude .env --exclude '.env.*' --exclude '.env-*' \
  --exclude '.env *' --exclude .envrc \
  --exclude '*credentials*.json' --exclude '*private*key*' \
  --exclude '*.key' --exclude '*.pem' --exclude '*.p12' --exclude '*.p8' \
  --exclude id_rsa --exclude id_dsa --exclude id_ecdsa --exclude id_ed25519 \
  ./ wu-ta-service:~/world-understanding/
ssh wu-ta-service "cd ~/world-understanding && docker compose -f apps/texture_agent_service/docker-compose.yml up -d --build texture-agent-service"
brev port-forward wu-ta-service -p 8001:8001
curl -fsS http://localhost:8001/health
```

## Output Format

Return the Brev instance names, endpoint URLs, selected model IDs, Texture
Agent environment exports, smoke-test result, service URL when used, and cleanup
commands.

## Troubleshooting

- `brev exec` may print `external nodes: skipping (list failed): not_found:
  not found` before a successful command. Treat it as non-fatal if the command
  exits zero.
- Brev images may make `/workspace` read-only for `ubuntu`; use
  `~/world-understanding`.
- Check `df -h /` before serving larger local models. Some images expose a
  smaller root filesystem than the requested disk size; failed Hugging Face
  downloads can fill it quickly. Delete the VM and try another provider/type if
  there is not a writable cache location with enough free disk.
- On AWS `g6e.xlarge`, root is small enough that Docker and FLUX NIM cache
  should be moved to `/opt/dlami/nvme` before pulling the image. The NIM cache
  mount must be writable by the container user.
- FLUX NIM readiness is `GET /v1/health/ready`. Do not use `/v1/models` as the
  readiness gate for this image-generation NIM; it returned 404 during
  validation.
- The FLUX image-generation NIM handled both `/v1/images/generations` and
  `/v1/images/edits`, which Texture Agent uses for albedo and conditioned PBR
  map generation through the OpenAI-compatible image-generation client.
- If using a noninteractive shell with `set -u`, keep Docker's NGC username as
  the literal string `$oauthtoken`; escape the `$` so it is not treated as a
  shell variable.
- Disable Qwen thinking for Texture Agent too; it expects direct LLM content,
  not reasoning traces in `message.content`.
- If Brev auth fails with `malformed refresh token` before deletion, power off
  reachable VMs over SSH to stop compute spend, then ask the user to re-login
  so `brev delete` can finish.
- After `brev delete`, a workspace can remain visible as `DELETING` with
  `shell_status: NOT READY` while the Brev control plane catches up. Retry
  `brev delete`, issue `brev stop` if supported, and poll `brev ls --json`
  until it disappears.
- Keep the Brev image-generation endpoint alive only while the local pipeline
  is running; it can dominate runtime and GPU spend compared with the Texture
  Agent LLM.

## Cleanup

```bash
brev delete wu-ta-service
brev delete wu-ta-image-gen
brev delete wu-ta-llm
brev ls --json
```
