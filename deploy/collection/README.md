# Content Agents Collection Deployment

This deployment path runs the Content Agents package without Helm and without a
single monolithic Compose stack.

The CPU-only agent services run together:

- Material Agent Service
- Physics Agent Service
- Texture Agent Service

Shared GPU/model dependencies are configured by endpoint:

- OVRTX rendering endpoint, required by material and physics.
- VLM endpoint, optional for material and physics.
- LLM endpoint, optional for material and texture.
- Image generation endpoint, optional for Texture Agent texture generation,
  Material Agent-generated reference images, and Material Agent-generated
  material-library textures.
- Embedding endpoint, optional for material prim clustering.

Brev is supported as an optional way to host any GPU dependency. It is not
required by the deployment model.

## Quickstart

1. Put API keys in the repo-root `.env` file.
2. Edit `deploy/collection/collection.yaml`.
3. Confirm the plan:

```bash
./deploy/collection/deploy.py plan
```

4. Start the CPU-only agents:

```bash
./deploy/collection/deploy.py up
```

On first run, `up` attempts a best-effort download of the public Scene Optimizer
Core bundle used by local Material/Physics `optimize_usd` flows. Docker Compose
still starts if that download is unavailable, which keeps remote optimizer or
optimization-disabled deployments usable. To make local Scene Optimizer resources
mandatory, run `up --require-local-scene-optimizer`.

5. Check health:

```bash
./deploy/collection/deploy.py status
```

Default local URLs:

- Material: http://localhost:8100/health
- Physics: http://localhost:8200/health
- Texture: http://localhost:8300/health

## Endpoint Model

The deployment spec is the source of truth:

```yaml
dependencies:
  render:
    enabled: true
    provider: external
    endpoint: http://host.docker.internal:8001

  vlm:
    enabled: false
    provider: external
    endpoint: ""
    backend: nim
    model: Qwen/Qwen2.5-VL-7B-Instruct
    max_tokens: 512
```

Use `provider: external` when the endpoint already exists. Use `provider:
local` when another local process or Compose stack provides the endpoint. Use
`provider: brev` only when the user wants Brev-hosted dependencies.
Set `max_tokens` when a smaller self-hosted model cannot accept the default
large completion budget.

## Examples

Reusable example specs live in `deploy/collection/examples/`:

- `minimal.yaml`: CPU-only agents plus a local or external OVRTX endpoint.
- `brev-render-only.yaml`: Brev-hosted OVRTX only.
- `brev-render-embeddings.yaml`: Brev-hosted OVRTX plus embedding NIM.
- `full-brev.yaml`: target shape for render, VLM/LLM, image-gen, and embeddings
  behind Brev forwards. The Brev VLM/LLM example uses the validated
  `Qwen/Qwen2.5-VL-7B-Instruct` vLLM endpoint on local port `8015`.

Run any example with:

```bash
./deploy/collection/deploy.py -c deploy/collection/examples/brev-render-embeddings.yaml plan
```

## Optional Brev Helper

When a spec uses `provider: brev`, the CLI can print the matching Brev
instances, create commands, forwards, and Docker-reachable endpoints:

```bash
./deploy/collection/deploy.py -c deploy/collection/examples/full-brev.yaml brev-plan
```

Creation is dry-run by default. To create only missing instances, explicitly
opt in:

```bash
./deploy/collection/deploy.py -c deploy/collection/examples/full-brev.yaml brev-provision --execute
```

Port forwards are printed but not started by `brev-provision` because they are
long-running foreground processes.

## Remote CPU Host

To stage the CPU agent deployment onto another Linux host, use the rsync helper.
It defaults to dry-run and excludes `.env`, virtualenvs, caches, git metadata,
and generated env files:

```bash
./deploy/collection/remote-rsync.sh user@cpu-host /opt/content-agents
./deploy/collection/remote-rsync.sh --execute user@cpu-host /opt/content-agents
```

## Shared OVRTX

Run OVRTX separately, then point `dependencies.render.endpoint` at it.

Same Docker host:

```bash
OVRTX_RENDER_MODE=pt docker compose \
  -f apps/ovrtx_rendering_api/docker-compose.yml up -d --build
```

Then use:

```yaml
endpoint: http://host.docker.internal:8001
```

Remote host or Brev port-forward:

```yaml
endpoint: http://render-host.example:8001
```

## Optional VLM / LLM

Material and Physics can use an OpenAI-compatible VLM endpoint. Material and
Texture can use an OpenAI-compatible LLM endpoint. For a local GPU host, run
the standalone vLLM roles outside the CPU agent stack:

```bash
docker compose -f deploy/collection/docker-compose.vlm.yml up -d
docker compose -f deploy/collection/docker-compose.llm.yml up -d
```

The VLM compose file uses the NVIDIA runtime path used by the current A100
Brev validation hosts. If the host exposes GPU access only through Docker
device requests, use the dedicated `deploy-qwen-vlm-brev` skill or a
provider-specific override. For hosts with a large mounted data disk, set
`COLLECTION_VLM_CACHE_VOLUME=/path/to/huggingface-cache` before startup.

Then enable the endpoints:

```yaml
dependencies:
  vlm:
    enabled: true
    provider: local
    endpoint: http://host.docker.internal:8015/v1
    backend: nim
    model: Qwen/Qwen2.5-VL-7B-Instruct
    max_tokens: 512
  llm:
    enabled: true
    provider: local
    endpoint: http://host.docker.internal:8017/v1
    backend: nim
    model: Qwen/Qwen2.5-7B-Instruct
    max_tokens: 512
```

The Brev validation path shares the Qwen VLM endpoint for LLM calls by default.

For OpenAI-compatible endpoints, set `backend: openai` and use `base_url`.
Use `api_key_env` when the secret should be resolved from an environment
variable inside the agent container, or `api_key` for an endpoint-scoped inline
placeholder such as a local no-auth gateway:

```yaml
dependencies:
  vlm:
    enabled: true
    provider: external
    backend: openai
    model: my-custom-vlm
    base_url: https://vlm-gateway.example/v1
    api_key_env: CUSTOM_VLM_API_KEY
  llm:
    enabled: true
    provider: external
    backend: openai
    model: my-custom-llm
    base_url: https://llm-gateway.example/v1
    api_key_env: CUSTOM_LLM_API_KEY
```

The names in `api_key_env` must be available to the agent services, for example
through the collection `.env` file used by Docker Compose.

## Optional Embeddings

Material Agent can use an optional embedding endpoint for prim clustering. Run
the standalone embedding NIM on a GPU host:

```bash
export COLLECTION_EMBEDDINGS_CACHE_VOLUME=/path/to/writable-nim-cache  # optional
docker compose -f deploy/collection/docker-compose.embeddings.yml up -d
```

Then enable the endpoint:

```yaml
dependencies:
  embeddings:
    enabled: true
    provider: external
    endpoint: http://host.docker.internal:8004/v1
    backend: nim
    model: nvidia/llama-nemotron-embed-vl-1b-v2
    api_key: not-used
```

For Brev-hosted embeddings, forward the remote port to a Docker-reachable local
port and use `host.docker.internal:<port>` from the agent containers.

## Optional Image Generation

Texture Agent texture generation, Material Agent-generated reference images,
and Material Agent-generated material-library textures can share an optional
OpenAI-compatible image-generation endpoint. Run the standalone FLUX NIM on a
GPU host:

```bash
export COLLECTION_IMAGE_GEN_CACHE_VOLUME=/path/to/writable-nim-cache  # optional
docker compose -f deploy/collection/docker-compose.image-gen.yml up -d
```

Then enable the endpoint:

```yaml
dependencies:
  image_gen:
    enabled: true
    provider: external
    endpoint: http://host.docker.internal:8005/v1
    backend: openai
    model: black-forest-labs/flux.2-klein-4b
    api_key: not-used
```

For Brev-hosted image generation, forward the remote port to a
Docker-reachable local port and use `host.docker.internal:<port>` from the
agent containers.

## Stop

```bash
./deploy/collection/deploy.py down
```
