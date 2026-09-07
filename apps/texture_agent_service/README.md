# Texture Agent Service

FastAPI service for AI-driven texture generation on materialized USD assets.
It wraps the [Texture Agent](../texture_agent/) pipeline with session
management, Server-Sent Events, artifact download routes, and an OpenAPI
contract.

> **Content Agents 0.6 routing:** this service is an explicit fixed-pipeline
> REST interface. For an unqualified texture task, start at the repository root
> with `content-workflow-texture` and its focused `content-workflow-cli texture
> prepare` through `texture publish` operations. Use this service when its stable
> HTTP, session, or deployment contract is specifically required.

## Backend Boundary

The default `simple_image_gen` path calls the configured hosted image-generation
provider. The optional `service` path calls a Texture Variation API-compatible
endpoint supplied by the operator:

```bash
TA_TEXTURE_BACKEND=service
TA_TEXTURE_ENDPOINT=https://texture-variation.example.test
TA_BACKEND_ENGINE=YOUR_ENGINE_OR_MODEL
```

This source release ships the client/adapter contract. It does not install,
download, or distribute Step1X, Material Anything, Swin2SR, their model assets,
or their runtime environments. Operators provision compatible backends
independently and are responsible for reviewing and complying with all upstream
license terms.

## Quick Start

Requires Docker Compose v2.24+.

```bash
# From the repository root. Keep real keys in an ignored .env file.
cp .env_example .env

docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml up --build
```

The service is available at `http://localhost:8001`. Its default stack is
CPU-only and uses the configured hosted image-generation and LLM providers.
Optional local NIM profiles are documented in
[`docker-compose.yml`](docker-compose.yml) and require an NVIDIA GPU.

All published Compose ports bind to `127.0.0.1` by default, including optional
VLM, image-generation, Texture Variation, and OVRTX sidecars. Set
`WU_COMPOSE_BIND_HOST` only when an explicit network access-control and
authentication boundary protects every published service.

To connect an operator-provided Texture Variation service, place the adapter
variables in the repository-root `.env` before starting Compose:

```dotenv
TA_TEXTURE_BACKEND=service
TA_TEXTURE_ENDPOINT=https://texture-variation.example.test
TA_BACKEND_ENGINE=YOUR_ENGINE_OR_MODEL
```

## API

- Interactive docs: `http://localhost:8001/docs`
- OpenAPI snapshot: [`openapi.yaml`](openapi.yaml)
- Detailed API guide: [`docs/api.md`](docs/api.md)
- Health: `GET /health`
- Submit: `POST /pipeline`
- Progress: `GET /pipeline/{session_id}/events`

The API accepts a materialized USD upload plus optional material/prim scope.
Use explicit scope for large assets and inspect the returned UV/backend
diagnostics before treating an output as valid.

## Development

```bash
source .venv/bin/activate
pytest apps/texture_agent_service/tests
```
