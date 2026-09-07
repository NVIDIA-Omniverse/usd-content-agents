---
name: deploy-image-gen-brev
description: "Fixed pipeline reference: Deploy or smoke-test a Brev-hosted FLUX image-generation NIM endpoint for Content Agents. Use when the user asks to deploy image generation on Brev, host FLUX.2 Klein for Texture Agent, enable image-gen sidecars, test image-gen endpoints, or create a Brev image-gen endpoint for the collection deployment."
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
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
compatibility: Requires Brev CLI access, Docker on the remote GPU instance, NGC_API_KEY and HF_TOKEN for FLUX NIM startup, and local port-forward access.
---

# Fixed Pipeline Reference: Deploy Image Generation With Brev

## When to Use

Use this skill for a standalone FLUX image-generation endpoint that local
Content Agent services can reach through an explicit URL.

The golden path mirrors the validated Texture Agent Brev flow:

- Brev type: `g6e.xlarge`
- Instance name: `content-image-gen`
- GPU: one L40S
- Image: `nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant`
- Remote host port: `8000`
- Local Docker-reachable forward: `0.0.0.0:8016 -> content-image-gen:8000`
- Collection endpoint: `http://host.docker.internal:8016/v1`

This endpoint is optional. Material, Physics, and Texture Agent services remain
CPU-only; this skill hosts only the image-generation sidecar.

## Limitations

- Keep NGC and Hugging Face credentials in `.env` or remote env files with
  restrictive permissions; never print or commit them.
- Use this as an optional image-generation dependency; keep agent services
  CPU-only.
- Delete the GPU node after validation unless the user asks to keep it.

## Prerequisites

- Brev CLI access and an SSH-ready `g6e.xlarge` or equivalent L40S GPU node.
- Docker with enough writable storage for the FLUX NIM image and model cache.
- `NGC_API_KEY` and `HF_TOKEN` available locally before copying the remote env.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Create or reuse the Brev node only after the dry-run looks acceptable.
3. Qualify storage, GPU, Docker root, and NIM cache before pulling FLUX.
4. Authenticate with NGC and Hugging Face, then start the FLUX NIM.
5. Forward the endpoint, run readiness and image-generation smokes, wire the
   collection config, and clean up as needed.

## Credit Safety

Start with inventory and a dry-run:

```bash
brev ls --json
brev create content-image-gen --dry-run --type g6e.xlarge --min-disk 500
```

Only after the instance type and spend are acceptable:

```bash
brev create content-image-gen --type g6e.xlarge --min-disk 500 --timeout 1200
```

Wait for shell readiness before copying credentials or starting containers.

## Qualify Storage And GPU

Use the AWS DLAMI NVMe path when it exists:

```bash
brev exec content-image-gen \
  "df -h / /opt/dlami/nvme /mnt/* 2>/dev/null || true; \
   lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; \
   docker info --format 'DockerRoot={{.DockerRootDir}}'; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
brev exec content-image-gen \
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
   docker info --format 'DockerRoot={{.DockerRootDir}}' && \
   df -h / /opt/dlami/nvme"
```

If the requested disk is not reflected in a writable filesystem, stop and
choose another provider/type before pulling the NIM image.

## NGC And HF Auth

FLUX NIM needs `NGC_API_KEY` for the container pull and `HF_TOKEN` for model
weights. Never print or commit them. Export both values from an approved secret
source before running this block. If they exist only in the repo `.env`, use
the repo planner, which parses only these keys without sourcing the file:

```bash
source .venv/bin/activate
python scripts/brev_agent_services.py --service texture --preset hybrid \
  --image-gen-node-name content-image-gen
```

Before sending credentials, verify the node's SSH host-key fingerprint through
an independently authenticated Brev/provider channel and install that exact
key in `~/.ssh/known_hosts`.

```bash
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
  if ! ssh "${ssh_options[@]}" content-image-gen true; then
    echo "A trusted SSH host key for content-image-gen is required before sending credentials" >&2
    exit 1
  fi
  printf "NGC_API_KEY=%s\nHF_TOKEN=%s\n" "$NGC_API_KEY" "$HF_TOKEN" | \
    ssh "${ssh_options[@]}" content-image-gen '
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
  ssh "${ssh_options[@]}" content-image-gen \
  "chmod 600 \"\$HOME/.ngc-nim.env\" && \
   NGC_API_KEY=\"\$(sed -n 's/^NGC_API_KEY=//p' \"\$HOME/.ngc-nim.env\")\" && \
   test -n \"\$NGC_API_KEY\" && \
   printf '%s\n' \"\$NGC_API_KEY\" | \
     docker login nvcr.io -u '\$oauthtoken' --password-stdin"
)
```

## Start FLUX

```bash
ssh content-image-gen \
  'docker rm -f flux-image-gen >/dev/null 2>&1 || true; \
   docker run -d --name flux-image-gen --gpus all --shm-size=32g -p 8000:8000 \
     --env-file "$HOME/.ngc-nim.env" \
     -e NIM_CACHE_PATH=/opt/nim/.cache \
     -v /opt/dlami/nvme/nim-cache:/opt/nim/.cache \
     nvcr.io/nim/black-forest-labs/flux.2-klein-4b:1.0.1-variant'
```

Use `GET /v1/health/ready` for readiness. Do not use `/v1/models`; this NIM can
return 404 there.

```bash
ssh content-image-gen 'curl -fsS http://localhost:8000/v1/health/ready'
```

## Local Port-Forward And Smoke

For Docker-hosted agent containers, bind the forward to all host interfaces:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 0.0.0.0:8016:localhost:8000 content-image-gen
```

Smoke from the local machine:

```bash
curl -fsS http://localhost:8016/v1/health/ready
curl -fsS http://localhost:8016/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"black-forest-labs/flux.2-klein-4b","prompt":"a simple red square texture tile","size":"512x512","n":1}'
```

Then set collection config:

```yaml
dependencies:
  image_gen:
    enabled: true
    provider: brev
    endpoint: http://host.docker.internal:8016/v1
    backend: openai
    model: black-forest-labs/flux.2-klein-4b
    api_key: not-used
```

## Output Format

Return the Brev instance name, remote and local endpoint URLs, model ID,
readiness and image-generation smoke status, collection wiring values, and the
cleanup command.

## Troubleshooting

- If readiness fails, report the container status and recent FLUX NIM logs.
- If `/v1/images/generations` fails, include the HTTP status, response summary,
  and whether credentials, model startup, or port-forwarding is the likely
  blocker.

## Cleanup

Stop local port-forward processes and delete the image-generation node after
the test unless the user explicitly wants to keep it:

```bash
brev delete content-image-gen
```

If Brev deletion is blocked but SSH still works, power off the instance while
control-plane cleanup is retried:

```bash
ssh content-image-gen 'sudo poweroff'
```
