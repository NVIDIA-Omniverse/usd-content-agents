# Material Agent Service

FastAPI service for VLM-based material assignment to 3D USD files. Wraps the [Material Agent](../material_agent/) pipeline behind a REST API with session management, async progress streaming, and Docker-ready deployment.

## Quick Start (Docker)

Requires **Docker Compose v2.24+** (for `env_file: required: false` support).

```bash
# From the repo root -- set your VLM provider key
# (NIM, OpenAI, Anthropic, or Gemini). The compose file reads .env at
# the repo root via env_file.
echo 'NVIDIA_API_KEY=your_key' > .env

# Build and run (pulls in OVRTX rendering as a sidecar). `--env-file .env`
# is required so that any `${VAR}` overrides in compose (e.g.
# `MA_VLM_BACKEND=openai`) read from the repo-root `.env`. Without it,
# Compose's variable substitution looks for `.env` next to the compose
# file (`apps/material_agent_service/.env`) and silently falls back to
# the built-in defaults.
docker compose --env-file .env \
  -f apps/material_agent_service/docker-compose.yml up --build

# Service available at http://localhost:8000
```

The bundled `ovrtx-rendering-api` sidecar has a cold-start GPU warm-up phase.
Expect `material-agent-service` to stay blocked for roughly 5 minutes until the
sidecar health check flips to `gpu_initialized=true`.

See [`docs/docker.md`](docs/docker.md) for full Docker / Docker Compose details, including multi-GPU and VLM-NIM sidecar profiles.

## Quick Start (Local Dev)

```bash
# From repo root
source .venv/bin/activate

# Install
uv pip install -e ".[dev,telemetry]"
uv pip install -e apps/material_agent -e apps/material_agent_service

# Configure
cp apps/material_agent_service/.env_example apps/material_agent_service/.env
# Edit .env to set NVIDIA_API_KEY (or another VLM provider key)

# Run
cd apps/material_agent_service
uvicorn service.main:app --reload --port 8000
```

## API

- **Interactive docs:** http://localhost:8000/docs (Swagger UI) once the service is running.
- **Full reference:** [`docs/api.md`](docs/api.md).
- **OpenAPI spec:** [`openapi.yaml`](openapi.yaml).

The pipeline endpoints (`POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, etc.) accept a USD file plus optional materials manifest and reference images, then run the multi-step material assignment pipeline. Stream real-time progress over SSE at `GET /pipeline/{id}/events`.

Set `enable_material_generation=true` on `POST /pipeline` to generate an
asset-specific material library from uploaded reference images or a
`generated_reference_id`. The service uses deployment-time `MA_IMAGE_GEN_*`
settings for both generated reference images and generated material-library
textures; users do not provide image-generation endpoint credentials per run.

## Configuration

Service configuration is loaded from environment variables at startup. See [`.env_example`](.env_example) for the full list. Key settings:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Required if using `nim` VLM backend |
| `OPENAI_API_KEY` | Required if using `openai` backend |
| `ANTHROPIC_API_KEY` | Required if using `anthropic` backend |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Required if using `gemini` backend |
| `MA_VLM_BACKEND` | Default: `nim` |
| `MA_VLM_MODEL` | Default: `qwen/qwen3.5-397b-a17b` |
| `MA_IMAGE_GEN_BACKEND` | Shared image-generation backend for generated reference images and generated material-library textures (default: `gemini`) |
| `MA_IMAGE_GEN_MODEL` | Optional shared image-generation model override |
| `MA_IMAGE_GEN_BASE_URL` | Optional shared image-generation API base URL |
| `MA_IMAGE_GEN_API_KEY` | Optional shared image-generation API key; use `not-used` only for explicit no-auth local endpoints |
| `MA_RENDERER_BACKEND` | Default: `remote` (resolves via `RENDER_ENDPOINT`) |
| `RENDER_ENDPOINT` | URL of OVRTX rendering API or compatible service |
| `MA_SESSION_STORAGE_PATH` | Where session directories are written |
| `MA_MAX_UPLOAD_SIZE_MB` | Max USD upload size (default: 500) |
| `MA_MAX_ACTIVE_SESSIONS` | Max concurrent pipelines. Source fallback: `3`; service image override: `8`; local Docker Compose override: `1`. Invalid or negative values fall back to `3`; `0` permits no active executions. |
| `MA_MAX_RENDER_NUM_WORKERS` | Max accepted render worker override. Service default: `32`; local Docker Compose default: `1` |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | Process-wide render request cap. Service default: unset/disabled; local Docker Compose default: `1` |

## Custom Materials

The service ships with a default material library at `materials/default/`. To use your own:

1. Create a subdirectory under `materials/` with a `materials.yaml` manifest (see existing default for the schema).
2. Restart the service — your library appears automatically in the model selector and at `GET /materials`.

The bundled default library does not ship preview thumbnails. If you want UI
icons for a custom library, add optional `thumbs/` images and reference them
from the `icon` fields in `materials.yaml`.

You can also upload a materials ZIP per-pipeline-call:

```bash
curl -X POST http://localhost:8000/pipeline \
  -F "usd_file=@scene.usd" \
  -F "materials_zip=@custom_materials.zip"
```

See [`examples/README.md`](examples/README.md) for the materials ZIP format.

Generated material-library runs write `generated_material_library/materials.yaml`,
`generated_material_library/material_library.usda`,
`generated_material_library/material_generation_plan.yaml`, and texture maps in
the session. If prediction returns `__UNKNOWN__` or an unrepaired invalid
material, the pipeline binds the canonical `__FALLBACK_MATERIAL__` neutral gray
matte opaque plastic material.

## Architecture

```
Upload USD → Session Created → Pipeline Runs → Download Output
                                    ↓
                            (SSE progress events)
                                    ↓
                            Preview images available
```

Pipeline steps run in order (configurable via env or per-request):

1. `optimize_usd` — Flatten/deinstance via scene optimizer
2. `build_dataset_usd` — Render prim views for VLM input
3. `build_dataset_prepare_dataset` — Compose dataset with material specs
4. `predict` — VLM inference for material assignment
5. `validate_predictions`, `harmonize_predictions` — Post-process
6. `apply` — Apply predictions back into the USD
7. `render` — Final render of materialized asset

## Development

```bash
# Run tests
pytest tests/

# Format and lint
./format.sh
```

## Project Structure

```
material_agent_service/
├── client/                     # Optional Python client
├── docs/                       # Documentation (api.md REST reference, docker.md deployment guide)
├── examples/                   # Custom materials ZIP example
├── materials/                  # Built-in material library + drop-in libraries
├── service/                    # FastAPI app, routers, runtime, storage
├── tests/                      # Test suite
├── docker-compose.yml          # Standard Docker Compose
├── docker-compose.multi-gpu.yml # Multi-GPU overlay
├── Dockerfile                  # Service image
├── openapi.yaml                # API specification
└── pyproject.toml              # Install metadata
```
