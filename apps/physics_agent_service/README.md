# Physics Agent Service

FastAPI service for VLM-based physics property classification of 3D USD assets. Wraps the [Physics Agent](../physics_agent/) pipeline behind a REST API with session management, async progress streaming, and Docker-ready deployment.

## Quick Start (Docker)

Requires **Docker Compose v2.24+** (for `env_file: required: false` support).

```bash
# From the repo root -- set your VLM provider key
# (NIM, OpenAI, Anthropic, or Gemini). The compose file reads .env at
# the repo root via env_file.
echo 'NVIDIA_API_KEY=your_key' > .env

# Choose one Scene Optimizer backend before starting the stack.
#
# Local bundle, used by the default `optimize_usd` path:
./scripts/fetch_build_resources.sh
#
# Remote NVCF optimizer instead of the local bundle:
# cat >> .env <<'EOF'
# NGC_API_KEY=your_ngc_key
# NVCF_OPTIMIZER_FUNCTION_ID=your_optimizer_function_id
# # or OPTIMIZER_ENDPOINT=https://...
# EOF

# Build and run (pulls in OVRTX rendering as a sidecar). `--env-file .env`
# is required so that any `${VAR}` overrides in compose (e.g.
# `PA_VLM_BACKEND=openai`) read from the repo-root `.env`. Without it,
# Compose's variable substitution looks for `.env` next to the compose
# file (`apps/physics_agent_service/.env`) and silently falls back to
# the built-in defaults.
docker compose --env-file .env \
  -f apps/physics_agent_service/docker-compose.yml up --build

# Service available at http://localhost:8000
```

`scripts/fetch_build_resources.sh` selects the Scene Optimizer Core package for
the host architecture when available. If the default package is not usable for
your platform, set `SO_CORE_URL` to an explicit Scene Optimizer Core zip, or skip
the local fetch and use a remote NVCF optimizer through `NGC_API_KEY` plus
`NVCF_OPTIMIZER_FUNCTION_ID` / `OPTIMIZER_ENDPOINT`.

To run with a local Cosmos VLM NIM sidecar on a second GPU:

```bash
docker login nvcr.io -u '$oauthtoken' -p $NGC_API_KEY
docker compose --env-file .env \
  -f apps/physics_agent_service/docker-compose.yml \
  -f apps/physics_agent_service/docker-compose.multi-gpu.yml \
  --profile vlm up --build
```

The multi-GPU overlay pins `ovrtx-rendering-api` to GPU 0 and `vlm-nim` to
GPU 1, and routes physics-agent VLM/LLM calls through `PA_VLM_NIM_BASE_URL`
and `PA_LLM_NIM_BASE_URL`.
Validate the sidecar pinning with:

```bash
docker exec physics-vlm-nim nvidia-smi --query-gpu=count --format=csv,noheader
# expected: 1
```

The bundled `ovrtx-rendering-api` sidecar has a cold-start GPU warm-up phase.
Expect `physics-agent-service` to stay blocked for roughly 5 minutes until the
sidecar health check flips to `gpu_initialized=true`.

## Quick Start (Local Dev)

```bash
# From repo root
source .venv/bin/activate

# Install
uv pip install -e ".[dev]"
uv pip install -e apps/physics_agent -e apps/physics_agent_service

# Configure
cp .env_example .env
# Edit .env to set NVIDIA_API_KEY (or another VLM provider key)

# Run
cd apps/physics_agent_service
uvicorn service.main:app --reload --port 8000
```

## API

- **Interactive docs:** http://localhost:8000/docs (Swagger UI) once the service is running.
- **Full reference:** [`docs/api.md`](docs/api.md).
- **Brev deployment planning:** [`docs/brev.md`](docs/brev.md).
- **OpenAPI spec:** [`openapi.yaml`](openapi.yaml).

The pipeline endpoints (`POST /pipeline`, `GET /pipeline/{id}/status`, etc.) accept a USD file — uploaded directly, referenced by S3 URI, or already staged in an existing session — then run the multi-step classification pipeline (optimize, identify asset, render, build dataset, predict, apply physics). `POST /pipeline` requires at least one source and resolves multiple source fields with `session_id > s3_uri > usd_file` precedence. Stream real-time progress over SSE at `GET /pipeline/{id}/events`.

The tune endpoints (`POST /tune`, `GET /tune/{id}/status`, `GET /tune/{id}/results`, `GET /tune/{id}/events`, `POST /tune/{id}/cancel`, `GET /tune/{id}/artifacts/{name}`) run BoTorch-first physics parameter tuning against a simulation-ready USD authored by `apply_physics`. The refine endpoints (`POST /refine`, `GET /refine/{id}/status`, `GET /refine/{id}/results`, `GET /refine/{id}/events`, `POST /refine/{id}/cancel`, `GET /refine/{id}/artifacts/{name}`) run the iterative tune-judge-scenario-refine loop. Both reuse the same session manager / job registry / SSE / artifact storage infrastructure as `/pipeline`. Production tuning/refine images require the optional `tuning` extra (`uv pip install -e "apps/physics_agent[tuning]"`) and an OvPhysX daemon venv; `Dockerfile` and `Dockerfile.ci` provision those for deployment images. Internal NVIDIA inference backend registration is supplied by the staged optional backend wheel or internal package in internal builds. The `/refine` worker builds server-side judge/refiner models from deployment configuration; clients do not submit model provider credentials through this route. See [`../physics_agent/docs/tuning.md`](../physics_agent/docs/tuning.md) for tuning architecture and extension points.

Deployment images support Linux `amd64` and `arm64` with reviewed,
architecture-specific tuning and OvPhysX locks. `GET /health` reports
`tuning_extra_available` and `ovphysx_runtime_available`; both should be `true`
for the default BoTorch + OvPhysX tuning path.

## Python Client

See [`client/README.md`](client/README.md) for the bundled Python client, which supports both local file upload and S3 URI input modes. Example:

```bash
# Local file upload
python apps/physics_agent_service/client/client.py /path/to/scene.usdz

# S3 URI (service downloads server-side)
python apps/physics_agent_service/client/client.py \
  --s3-uri s3://your-bucket/path/to/scene.usdz
```

## Configuration

Service configuration is loaded from environment variables at startup. Key settings:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Required if using `nim` VLM backend |
| `OPENAI_API_KEY` | Required if using `openai` backend |
| `ANTHROPIC_API_KEY` | Required if using `anthropic` backend |
| `GOOGLE_API_KEY` | Required if using `gemini` backend |
| `PA_VLM_BACKEND` | Default: `nim` |
| `PA_VLM_MODEL` | Default: `qwen/qwen3.5-397b-a17b` |
| `PA_TUNE_BACKEND` | Optional `/tune` prompt interpreter backend override; falls back to `PA_REFINE_BACKEND`, then `PA_VLM_BACKEND` |
| `PA_TUNE_MODEL` | Optional `/tune` prompt interpreter model override; falls back to `PA_REFINE_MODEL`, then `PA_VLM_MODEL` |
| `PA_REFINE_BACKEND` | Optional `/refine` judge/refiner backend override; falls back to `PA_VLM_BACKEND` |
| `PA_REFINE_MODEL` | Optional `/refine` judge/refiner model override; falls back to `PA_VLM_MODEL` or the deployment default |
| `PA_VLM_NIM_BASE_URL` | Optional local/custom NIM endpoint for physics VLM calls |
| `PA_LLM_NIM_BASE_URL` | Optional local/custom NIM endpoint for physics LLM calls |
| `PA_NIM_API_KEY` | Endpoint-scoped NIM key, or `not-used` for a no-auth local sidecar |
| `NGC_API_KEY` | Required when using an NVCF Scene Optimizer backend |
| `NVCF_OPTIMIZER_FUNCTION_ID` | Optional remote NVCF Scene Optimizer function ID for `optimize_usd` |
| `OPTIMIZER_ENDPOINT` | Optional remote optimizer endpoint URL for `optimize_usd` |
| `PA_RENDER_BACKEND` | Default: `remote` (resolves via `RENDER_ENDPOINT`) |
| `RENDER_ENDPOINT` | URL of OVRTX rendering API or compatible service |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | Process-wide render request cap; local OVRTX compose defaults to `1` |
| `PA_SESSION_STORAGE_PATH` | Where session directories are written |
| `PA_MAX_UPLOAD_SIZE_MB` | Max USD upload size (default: 500) |
| `PA_S3_ALLOWED_BUCKETS` | Exact bucket names allowed for client-supplied `s3_uri` inputs, separated by commas or whitespace; empty/unset rejects all S3 URI inputs |

`PA_S3_ALLOWED_BUCKETS` is an application request-authorization policy for the
`pipeline`, `predict`, `tune`, and `refine` routes. Entries are bucket names
only, not `s3://` URIs, wildcards, or key prefixes; this release does not add
application-level key-prefix restrictions. A missing allowlist or a bucket
that is not an exact match returns the same generic HTTP `403` before any
`HeadObject` or download. File uploads and existing-session inputs are
unaffected.

Before upgrading a deployment whose clients use `s3_uri`, set the allowlist or
migrate those clients to file uploads or existing-session inputs. Leaving it
unset intentionally changes all S3 URI requests to `403`; restart the service
after changing the setting.
The Helm chart exposes the same setting as `s3AllowedBuckets`.

Use a dedicated intake bucket because every key in an allowed bucket is in
application scope. Restrict the service IAM role to only the required bucket
and prefixes as defense in depth; IAM scoping is not a substitute for the
allowlist.

For the public Docker deployment, configure these backend variables with one
of the documented public providers and pass credentials through the provider's
environment variable or an endpoint-scoped `PA_NIM_API_KEY`.

## Architecture

```
Upload USD → Session Created → Pipeline Runs → Download Output
                                    ↓
                            (SSE progress events)
                                    ↓
                            Per-component classification
```

Pipeline steps run in order:

1. `optimize_usd` — Flatten/deinstance via scene optimizer when enabled (`optimize_usd`, `enable_deinstance`, `enable_split`, `enable_deduplicate` form flags)
2. `identify_asset` — Preview-render whole asset, VLM identifies asset type
3. `build_dataset_usd` — Render per-prim views for VLM input
4. `build_dataset_prepare_dataset` — Compose dataset with classification specs
5. `predict` — VLM inference for per-component classification (type, material, physics)
6. `restore_usd` — Map optimized prediction paths back to original paths when `optimize_usd` is enabled.
7. `apply_physics` — Author `UsdPhysics.RigidBodyAPI` / `CollisionAPI` / `MassAPI` / `MaterialAPI` on each predicted prim plus a `PhysicsScene`. The output preserves `.usd`, `.usda`, and `.usdc` input extensions; USDZ inputs default to USDA output so Omniverse MDL shader references remain as runtime-resolved asset paths instead of being bundled into a new USDZ package. When optimization ran, physics is authored on the optimized/deinstanced USD so instance-proxy descendants are writable. Downloadable via `GET /artifacts/{id}/output-usd`.

## Project Structure

```
physics_agent_service/
├── client/                     # Python client (file-upload + S3 modes)
├── docs/                       # Documentation (api.md REST reference)
├── service/                    # FastAPI app, routers, runtime, storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker Compose (service + OVRTX sidecar)
├── Dockerfile                  # Service image
├── openapi.yaml                # API specification
└── pyproject.toml              # Install metadata
```
