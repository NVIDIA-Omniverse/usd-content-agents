---
name: deploy-embeddings-brev
description: "Fixed pipeline reference: Deploy or smoke-test a Brev-hosted NVIDIA embedding NIM endpoint for Content Agents. Use when the user asks to deploy embeddings on Brev, host the llama-nemotron-embed-vl model, enable Material Agent prim clustering embeddings, test embedding sidecars, or create a Brev embedding endpoint for the collection deployment."
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - brev
  - embeddings
  - deployment
tools:
  - Shell
  - Docker
  - Python
  - curl
  - Filesystem
  - brev
  - ssh
compatibility: Requires Brev CLI access, Docker on the remote GPU instance, NGC_API_KEY for NIM pulls, optional NVIDIA_API_KEY for hosted endpoint tests, and local port-forward access.
---

# Fixed Pipeline Reference: Deploy Embeddings With Brev

## When to Use

Use this skill for a standalone embedding endpoint that local Content Agent
services can reach through an explicit URL.

The golden path mirrors the validated OVRTX Brev path:

- Brev type: `g6e.xlarge`
- Instance name: `content-embeddings`
- GPU: one L40S
- Image: `nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0`
- Remote host port: `8004`
- Local Docker-reachable forward: `0.0.0.0:8014 -> content-embeddings:8004`
- Collection endpoint: `http://host.docker.internal:8014/v1`

Use a separate Brev instance from OVRTX unless the user explicitly asks for a
single-node fallback. Keep agents CPU-only; this skill only hosts the embedding
model sidecar.

## Limitations

- Keep NGC and NVIDIA credentials in `.env` or remote env files with restrictive
  permissions; never print or commit them.
- Use a separate Brev instance from OVRTX unless the user explicitly asks for a
  single-node fallback.
- Delete the GPU node after validation unless the user asks to keep it.

## Prerequisites

- Brev CLI access and an SSH-ready `g6e.xlarge` or equivalent L40S GPU node.
- Docker with enough writable storage for the embedding NIM image and cache.
- `NGC_API_KEY` for the container pull and optional `NVIDIA_API_KEY` for hosted
  endpoint compatibility checks.

## Instructions

1. Use `brev-cli` for generic Brev inventory, dry-run, create, port-forward,
   stop, delete, and cleanup guardrails.
2. Create or reuse the Brev node only after the dry-run looks acceptable.
3. Qualify storage, GPU, and Docker cache location before pulling NIM images.
4. Copy non-secret repo contents, authenticate to NGC, and start the NIM.
5. Forward the endpoint, run readiness and embedding smoke tests, wire the
   collection config, and clean up.

## Credit Safety

Start with inventory and a dry-run:

```bash
brev ls --json
brev create content-embeddings --dry-run --type g6e.xlarge --min-disk 500
```

Only after the instance type and spend are acceptable:

```bash
brev create content-embeddings --type g6e.xlarge --min-disk 500 --timeout 1200
```

Wait for shell readiness before copying files or starting containers.

## Qualify Storage And GPU

Check that Docker has enough writable disk for the NIM image and model cache:

```bash
brev exec content-embeddings \
  "df -h / /opt/dlami/nvme /mnt/* 2>/dev/null || true; \
   lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; \
   docker info --format 'DockerRoot={{.DockerRootDir}}'; \
   nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader"
```

On the validated AWS DLAMI-style image, use the ephemeral NVMe for Docker:

```bash
brev exec content-embeddings \
  "set -e; \
   sudo mkdir -p /opt/dlami/nvme/docker && \
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
   docker info --format 'DockerRoot={{.DockerRootDir}}'"
```

If the requested disk is not reflected in any writable filesystem, stop and
choose another provider/type before pulling the NIM image.

## Copy The Repo

Copy only non-secret repo contents. Prefer a committed `git archive` when the
embedding Compose file already exists on the current branch:

```bash
git archive --format=tar.gz -o /tmp/content-agents-collection.tar.gz HEAD
ssh content-embeddings 'rm -rf ~/collection-deployment && mkdir -p ~/collection-deployment'
scp /tmp/content-agents-collection.tar.gz \
  content-embeddings:~/collection-deployment/
ssh content-embeddings \
  'cd ~/collection-deployment && \
   tar -xzf content-agents-collection.tar.gz && \
   rm content-agents-collection.tar.gz'
```

For uncommitted testing, use `rsync` with the same secret/cache exclusions used
by the Brev agent skills.

## NGC Auth

Never print or commit secrets. Export `NGC_API_KEY` and the optional
`NVIDIA_API_KEY` from an approved secret source before this block; never source
the repo `.env` as shell code. Verify the node's SSH host-key fingerprint
through an independently authenticated Brev/provider channel and install that
exact key in `~/.ssh/known_hosts` before sending credentials.

```bash
(
  set -e
  if [ -z "${NGC_API_KEY:-}" ]; then
    echo "NGC_API_KEY is required" >&2
    exit 1
  fi
  NVIDIA_API_KEY="${NVIDIA_API_KEY:-}"
  for credential_name in NGC_API_KEY NVIDIA_API_KEY; do
    credential_value="${!credential_name}"
    cleaned="$(printf %s "$credential_value" | tr -d "\r\n")"
    if [ "$cleaned" != "$credential_value" ]; then
      echo "$credential_name must be a single-line value" >&2
      exit 1
    fi
  done
  ssh_options=(
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o StrictHostKeyChecking=yes
    -o "UserKnownHostsFile=$HOME/.ssh/known_hosts"
  )
  if ! ssh "${ssh_options[@]}" content-embeddings true; then
    echo "A trusted SSH host key for content-embeddings is required before sending credentials" >&2
    exit 1
  fi
  printf 'NGC_API_KEY=%s\nNVIDIA_API_KEY=%s\n' "$NGC_API_KEY" "$NVIDIA_API_KEY" | \
    ssh "${ssh_options[@]}" content-embeddings '
      set -e
      umask 077
      destination="$HOME/collection-deployment/.env"
      tmp_env="$(mktemp "${destination}.XXXXXX")"
      cleanup() { rm -f -- "$tmp_env"; }
      trap cleanup EXIT
      trap "exit 1" HUP INT TERM
      cat > "$tmp_env"
      chmod 600 "$tmp_env"
      mv -f -- "$tmp_env" "$destination"
      trap - EXIT HUP INT TERM
    '
  printf '%s\n' "$NGC_API_KEY" | \
    ssh "${ssh_options[@]}" content-embeddings \
      'docker login nvcr.io -u \$oauthtoken --password-stdin'
)
```

`NVIDIA_API_KEY` is optional for local no-auth NIM use, but copying it is useful
when the same remote `.env` is reused for hosted endpoint tests.

## Start The Embedding NIM

```bash
ssh content-embeddings \
  'cd ~/collection-deployment && \
   COLLECTION_EMBEDDINGS_PORT=8004 \
   docker compose -f deploy/collection/docker-compose.embeddings.yml up -d'
```

Poll readiness:

```bash
ssh content-embeddings \
  'for i in $(seq 1 80); do \
     echo attempt=$i status=$(docker inspect -f "{{.State.Health.Status}}" content-embeddings-nim 2>/dev/null || echo missing); \
     curl -fsS http://localhost:8004/v1/health/ready && exit 0; \
     docker logs --tail 30 content-embeddings-nim || true; \
     sleep 30; \
   done; exit 1'
```

Check model listing:

```bash
ssh content-embeddings 'curl -fsS http://localhost:8004/v1/models'
```

## Local Forward And Smoke

For local CLI-only tests, `brev port-forward` is sufficient:

```bash
brev port-forward content-embeddings -p 8014:8004
curl -fsS http://localhost:8014/v1/health/ready
```

For local Docker agent containers, bind the SSH forward to all host interfaces
so `host.docker.internal` can reach it:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 0.0.0.0:8014:localhost:8004 content-embeddings
```

Smoke image embedding inference:

```bash
python3 - <<'PY'
import json
import urllib.request

payload = {
    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
    "input": [
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0l"
        "EQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    ],
    "encoding_format": "float",
    "modality": ["image"],
    "input_type": "passage",
    "truncate": "NONE",
}
request = urllib.request.Request(
    "http://localhost:8014/v1/embeddings",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=60) as response:
    data = json.loads(response.read().decode("utf-8"))
print({"embedding_dim": len(data["data"][0]["embedding"])})
PY
```

Expected embedding dimension: `2048`.

## Collection Wiring

In `deploy/collection/collection.yaml` or an ignored test config:

```yaml
dependencies:
  embeddings:
    enabled: true
    provider: brev
    endpoint: http://host.docker.internal:8014/v1
    backend: nim
    model: nvidia/llama-nemotron-embed-vl-1b-v2
    api_key: not-used
```

Then run:

```bash
./deploy/collection/deploy.py -c deploy/collection/.brev-test.yaml plan
./deploy/collection/deploy.py -c deploy/collection/.brev-test.yaml up
./deploy/collection/deploy.py -c deploy/collection/.brev-test.yaml smoke
docker exec content-material-agent-service sh -lc \
  'curl -fsS "$MA_CLUSTER_EMBEDDING_BASE_URL/health/ready"'
```

## Output Format

Return the Brev instance name, remote and local endpoint URLs, model ID, smoke
test result, collection wiring values, and cleanup command.

## Troubleshooting

- If readiness fails, report the container health status and recent NIM logs.
- If embedding inference fails, include the HTTP status, response body summary,
  and whether the port-forward or model service is the likely blocker.

## Cleanup

Stop local port-forward processes and delete the embedding node after the test
unless the user explicitly wants to keep it:

```bash
brev delete content-embeddings
```

If Brev deletion is blocked but SSH still works, power off the instance to stop
compute spend while control-plane cleanup is retried:

```bash
ssh content-embeddings 'sudo poweroff'
```
