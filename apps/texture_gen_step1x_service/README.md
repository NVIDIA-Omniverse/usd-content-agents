# Texture Variation API Adapter Service

Optional adapter service for a Step1X-compatible Texture Variation API
backend. The package owns the FastAPI request/response contract, validation,
health reporting, job lifecycle, and translation to an external runtime
command.

## Distribution Boundary

This public source release does not install, download, or distribute Step1X,
Material Anything, Swin2SR, their model assets, or their Python/native runtime
environments. An operator must provision a compatible runtime and model cache,
mount them into the service, and review all applicable upstream license terms.

NVIDIA-managed runtime definitions, dependency locks, component inventories,
model inventories, and source-hydration metadata remain internal and are not
included in the public source release. The public package contains only the
adapter and its operator-mounted runtime contract.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/texture-variations` | Submit a texture generation job |
| `GET` | `/v1/texture-variations/{job_id}` | Query job status |
| `DELETE` | `/v1/texture-variations/{job_id}` | Cancel a job |
| `GET` | `/health` | Inspect adapter and mounted-runtime readiness |

The adapter boundary is `Step1XRunner` in `backend.py`. The default external
runner invokes an `edit_texture.py`-compatible command in a separate process;
the FastAPI service does not import or initialize the model runtime at startup.

## Operator-Mounted Runtime

At minimum, the service container needs a mounted runtime, its Python
environment, and any model/cache paths used by that environment:

| Variable | Description |
|---|---|
| `TEXTURE_STEP1X_RUNTIME_DIR` | Container path to the operator-provided runtime |
| `TEXTURE_STEP1X_EDIT_SCRIPT` | External edit script; defaults under the runtime directory |
| `TEXTURE_STEP1X_PYTHON` | Python executable inside the mounted runtime |
| `TEXTURE_STEP1X_MODEL_DIR` | Optional mounted model/cache root |
| `TEXTURE_STEP1X_CACHE_DIR` | Writable runtime cache path |
| `TEXTURE_OUTPUT_DIR` | Writable generated-artifact directory |
| `TEXTURE_STEP1X_MAX_WORKERS` | Concurrent external jobs; default `1` |
| `TEXTURE_STEP1X_TIMEOUT_SEC` | External command timeout; default `3600` |

The standalone Compose file wires those mounts without hydrating them:

```bash
TEXTURE_STEP1X_HOST_RUNTIME=/absolute/path/to/operator-runtime \
TEXTURE_STEP1X_HOST_MODEL_ROOT=/absolute/path/to/operator-model-cache \
TEXTURE_STEP1X_PYTHON=/opt/texture-editing/.venv/bin/python \
TEXTURE_SHARED_HOST_ROOT=/absolute/path/to/shared-work \
docker compose \
  -f apps/texture_gen_step1x_service/docker-compose.yml \
  up --build
```

The published port binds to `127.0.0.1` by default. Set
`WU_COMPOSE_BIND_HOST` only behind an explicit network access-control boundary;
the adapter accepts caller-selected local paths and is not a public endpoint.

Readiness is available at `GET /health`. A live API process is not evidence
that the mounted runtime can execute an inference request; require
`ready=true` and run an operator-approved smoke request before production use.

For development without Docker:

```bash
uvicorn apps.texture_gen_step1x_service.app:app --port 8000
```
