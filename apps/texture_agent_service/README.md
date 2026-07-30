# Texture Agent Service

FastAPI service for AI-driven texture generation on materialized USD assets. Wraps the [Texture Agent](../texture_agent/) pipeline behind a REST API with session management, async progress streaming via Server-Sent Events (SSE), and Docker-ready deployment.

## Default backends (product-default change)

The service ships with an NVIDIA-first default stack:

| Role | Default | Env var | Model |
|---|---|---|---|
| Image generation | `nim` | `TA_IMAGE_GEN_BACKEND` | `black-forest-labs/flux_2-klein-4b` (build.nvidia.com) |
| Auto-prompt LLM  | `nim` | `TA_LLM_BACKEND`       | `google/gemma-4-31b-it` (build.nvidia.com) |

Both honor `NVIDIA_API_KEY`. One key unlocks the whole default path.

> **PBR coherence trade-off — read before deploying.**
>
> The cloud `nim` image-gen endpoint does not accept reference images, so
> the normal- and roughness-map passes (which otherwise condition on the
> generated albedo) run text-only on the default path. The pipeline still
> produces a full PBR set, but the normal/roughness maps are less coherent
> with the albedo than on a conditioning-capable backend. The pipeline
> logs one warning per run to the service stdout (visible via
> `docker logs texture-agent-service`) so operators can tell at a glance
> whether a given run went through the text-only path.
>
> To keep full PBR coherence:
> - run `docker compose --profile image-gen` (local FLUX.2 NIM sidecar —
>   same model, but exposes `images.edit` and supports conditioning), **or**
> - set `TA_IMAGE_GEN_BACKEND=gemini` or `TA_IMAGE_GEN_BACKEND=openai`
>   (both support img2img, but leave the NVIDIA-only stack).
>
> See the [root README](../../README.md) system requirements and
> [`../texture_agent/examples/`](../texture_agent/examples/) for the runnable
> simple image-gen example.

## Quick Start (Docker)

Requires **Docker Compose v2.24+** (for `env_file: required: false` support).

```bash
# From the repo root -- set your image-gen provider key
# (NIM or Gemini). The compose file reads .env at the repo root
# via env_file.
echo 'NVIDIA_API_KEY=your_key' > .env

# Build and run. `--env-file .env` is required so that any `${VAR}`
# overrides in compose (e.g. `TA_IMAGE_GEN_BACKEND=gemini`) read from
# the repo-root `.env`. Without it, Compose's variable substitution
# looks for `.env` next to the compose file
# (`apps/texture_agent_service/.env`) and silently falls back to the
# built-in defaults.
docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml up --build

# Service available at http://localhost:8001
```

By default the texture service does not start GPU sidecars; texture generation
runs against the configured image-gen backend and cold start is fast.

### Texture Variation API Backend

The public 0.5 source release ships the service backend contract and the
Step1X-compatible adapter, but does not ship a managed Step1X runtime,
downloader/setup package, model checkpoints, or runtime cache layout. To use a
Texture Variation API-compatible backend, deploy that backend separately after
your own security and legal review, then configure Texture Agent to call it:

```bash
TA_TEXTURE_BACKEND=service \
TA_TEXTURE_ENDPOINT=http://texture-variation-backend:8000 \
TA_BACKEND_ENGINE=step1x \
docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml \
  up --build
```

Use the optional `docker-compose.step1x.yml` overlay only when you have already
mounted a complete external runtime for `texture-gen-step1x`:

```bash
TEXTURE_STEP1X_HOST_RUNTIME=/path/to/reviewed/texture-editing-runtime \
docker compose --env-file .env \
  -f apps/texture_agent_service/docker-compose.yml \
  -f apps/texture_agent_service/docker-compose.step1x.yml \
  up --build
```

That overlay starts the API adapter and OVRTX sidecar, routes
`TA_TEXTURE_ENDPOINT=http://texture-gen-step1x:8000`, and requires the runtime
to be supplied through `TEXTURE_STEP1X_HOST_RUNTIME`. It does not download or
install Step1X, Material Anything, Swin2SR, Kaolin, nvdiffrast, or model
checkpoints.

## Quick Start (Local Dev)

```bash
# From repo root
source .venv/bin/activate

# Install
uv pip install -e ".[dev]"
uv pip install -e apps/texture_agent -e apps/texture_agent_service

# Configure
cp .env_example .env
# Edit .env to set NVIDIA_API_KEY or GOOGLE_API_KEY

# Run
texture-agent-service
# or: uvicorn service.main:app --host 0.0.0.0 --port 8001
```

## API

- **Interactive docs:** http://localhost:8001/docs (Swagger UI) once the service is running.
- **Full reference:** [`docs/api.md`](docs/api.md).
- **Brev deployment planning:** [`docs/brev.md`](docs/brev.md).
- **OpenAPI spec:** [`openapi.yaml`](openapi.yaml).

The pipeline endpoints (`POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, etc.) accept a materialized USD file (typically the output of the Material Agent) and a per-material texture prompt map, then run the texture discovery / generation / apply pipeline. Stream real-time progress over SSE at `GET /pipeline/{id}/events`. Download textured output USDZ, textures, and the run manifest via `/artifacts/{id}/output`, `/artifacts/{id}/textures`, and `/artifacts/{id}/manifest`.

For release validation, the service path should agree with the CLI path on the
primary ladder fixture: four discovered materials, one generated texture set
when only `Aluminum_Matte` is provided in `material_textures_json` with
`auto_prompt_enabled=false`, matching output USD counts, and the same UV/report
diagnostics. Omitting `auto_prompt_enabled` preserves the service's legacy
auto-prompting behavior. The repeatable fake-backend smoke test is
`apps/texture_agent_service/tests/unit/test_issue31_validation_smoke.py`;
real NIM/Gemini service runs should be recorded as manual evidence, not as the
only gate.

### Session Cleanup

Long-lived deployments should delete sessions after downloading required artifacts so session storage does not grow indefinitely:

```bash
curl -X DELETE http://localhost:8001/sessions/$SESSION_ID
```

`DELETE /sessions/{session_id}` returns `204 No Content` when the session, stored artifacts, and in-memory progress state are removed. It returns JSON `404 Not Found` when the session does not exist, and JSON `409 Conflict` when a live pipeline job is still active or a worker lock shows artifact writes are still in progress; cancel the pipeline and wait for the worker to stop before deleting it. If a service restart leaves a persisted `cancelling` status with no live worker lock, deletion is allowed so stale artifacts can be cleaned up.

### Artifact Response Types

The `/artifacts/{session_id}/...` routes use per-kind response media types:

| Endpoint | Success media type | Payload |
|----------|--------------------|---------|
| `GET /artifacts/{session_id}/materials` | `application/json` | Discovered material metadata |
| `GET /artifacts/{session_id}/manifest` | `application/json` | Schema-versioned artifact manifest |
| `GET /artifacts/{session_id}/textures` | `application/zip` | ZIP containing generated textures under `textures/` |
| `GET /artifacts/{session_id}/textures/{filename}` | `image/png` | Single texture image |
| `GET /artifacts/{session_id}/output` | `model/vnd.usdz+zip` | Self-contained textured USDZ |
| `GET /artifacts/{session_id}/renders` | `application/zip` | ZIP containing final rendered images under `renders/` |
| `GET /artifacts/{session_id}/renders/{filename}` | `image/png` | Single render image |
| `GET /artifacts/{session_id}/preview/{filename}` | `image/png` | Single material preview image |

Error responses, including missing artifacts, are JSON.

## Python Client

```python
from client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

# Upload and run
session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    material_textures={
        "Steel_Carbon": {
            "prompt": "rusted steel",
            "opacity": 0.85,
        },
    },
    auto_prompt_enabled=False,  # strict material_textures scope
)

# Download artifacts
client.download_output(session_id, "output.usdz")
client.download_textures(session_id, "./textures/")

# Delete the session after required artifacts are downloaded
client.delete_session(session_id)
```

The bundled CLI exits with status `0` only after the final pipeline status is
`completed`. Scripted callers that intentionally stop client polling early
should keep the printed session ID and poll `GET /pipeline/{id}/status` before
deciding whether a nonzero client exit is a hard failure.

Projection backend runs use the same client helper with the `service` texture
backend and an endpoint. Global conditioning is merged with material-specific
conditioning before each Issue #116 texture variation request:

```python
session_id, status = client.run_and_monitor(
    usd_path="ladder.usd",
    material_textures={
        "Aluminum_Matte": {
            "prompt": "matte aluminum",
            "reference_image_uris": ["file:///refs/aluminum.png"],
        },
    },
    auto_prompt_enabled=False,
    texture_backend="service",
    texture_endpoint="http://localhost:8011",
    backend_engine="fake_projection",
    backend_custom_parameters={"variant": "success_full_pbr"},
    reference_image_uris=["file:///refs/ladder.png"],
    reference_image_path="reference.png",
    multiview_image_uris=["file:///refs/view0.png"],
    seed=11631,
    strength=0.8,
    strict_scope=True,
)
```

For AOI, CAD, PCB, or SimReady-style assets where traces, vias, pads, labels,
holes, seams, components, or markings already exist as modeled geometry, pass
`detail_policy="surface_only"` globally or set `detail_policy` inside a material
or per-prim entry in `material_textures`. This adds conservative prompt
conditioning for simple image generation and passes policy metadata to
projection backends.

For the command-line client wrapper, pass `--disable-auto-prompt` to send
`auto_prompt_enabled=false` and keep the run scoped to `material_textures`.
Use `--detail-policy surface_only` for the global conservative/AOI policy.

## Configuration

Service configuration is loaded from environment variables at startup. Key
settings are below. Docker Compose packages may set topology-specific overrides
such as external Texture Variation API or OVRTX sidecar endpoints.

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_API_KEY` | - | API key for NIM image generation |
| `GOOGLE_API_KEY` | - | API key for Gemini image generation |
| `TA_TEXTURE_BACKEND` | `simple_image_gen` | Texture gen backend |
| `TA_TEXTURE_ENDPOINT` | - | Default Texture Variation API endpoint when `TA_TEXTURE_BACKEND=service` |
| `TA_BACKEND_ENGINE` | - | Default Texture Variation API engine/model hint, e.g. `step1x` |
| `TA_IMAGE_GEN_BACKEND` | `nim` | Image gen backend (`nim`, `gemini`, `openai`) |
| `TA_IMAGE_GEN_BASE_URL` | - | Override image-gen base URL; used by the multi-gpu overlay to route to the local FLUX sidecar |
| `TA_IMAGE_GEN_API_KEY` | - | Endpoint-specific image-gen key; the local NIM overlay sets `not-used` in Compose |
| `TA_TEXTURE_SIZE` | `1024` | Texture resolution |
| `TA_TEXTURE_WORKERS` | `4` | Parallel gen workers |
| `TA_TEXTURE_JOB_TIMEOUT_SEC` | `3600` | Per-material service backend wait timeout; raise for slow Step1X GPU jobs |
| `TA_AUTO_PROMPT_MAX_GENERATED_MATERIALS` | `64` | Maximum missing materials auto-prompt may select before requiring explicit scope; `0` disables the guard |
| `TA_TEXTURE_PLAN_DEFAULT_CAP` | `32` | Generic/simple-image-gen planning default from `texture-agent-plan.v1` |
| `TA_TEXTURE_PLAN_UV_AWARE_DEFAULT_CAP` | `16` | UV-aware/service and Step1X planning default from `texture-agent-plan.v1` |
| `TA_TEXTURE_PLAN_HARD_CAP` | `64` | Immutable planning maximum; plans above it require consolidation or narrower scope before backend work |
| `TA_MAX_TEXTURE_UNITS` | `64` | Compatibility executor guard from issue #463; keep aligned with the planning hard cap. It is not the normal selection default |
| `TA_BLEND_OPACITY` | `0.85` | Default blend opacity |
| `TA_UV_POLICY` | `generate_missing` | UV prep policy for service-created pipelines (`generate_missing`, `force_projection`, etc.) |
| `TA_UV_SCOPE` | `stage` | UV preparation scope; use `target_prims` only with explicit prim targets |
| `TA_UV_BACKEND` | `python` | UV prep backend |
| `TA_UV_PROJECTION` | `box` | Projection mode used when UVs are generated |
| `TA_UV_OVERWRITE_EXISTING` | `false` | Whether generated UVs overwrite existing UV primvars |
| `TA_UV_REBAKE_SOURCE_ALBEDO` | `false` | Rebake source albedo when scoped projection UVs are generated |
| `TA_UV_REBAKE_SIZE` | - | Optional source texture rebake resolution |
| `TA_UV_NORMALIZE_OUT_OF_RANGE` | `false` | Whether out-of-range UVs are normalized during UV prep |
| `TA_RENDER_ENABLED` | `false` | Enable final render step for service-created pipelines |
| `RENDER_ENDPOINT` | - | Remote renderer endpoint used by final render tasks |
| `TEXTURE_STEP1X_HOST_RUNTIME` | - | Host runtime directory for the optional external Step1X-compatible adapter overlay |
| `TEXTURE_GEN_SIMPLE_ENV_FILE` | `/dev/null` | Optional extra env file for the simple sidecar, useful when provider keys live outside this checkout |
| `TEXTURE_STEP1X_GPU_DEVICE` | `0` | GPU device ID for the Step1X sidecar |
| `OVRTX_GPU_DEVICE` | `1` | GPU device ID for the OVRTX sidecar; set both GPU vars to `0` on one-GPU hosts |
| `TEXTURE_STEP1X_PYTHON` | - | Advanced override for custom mounted runtimes |
| `TEXTURE_STEP1X_LD_LIBRARY_PATH` | reference `.venv_gen` torch/CUDA libs | Advanced override when a custom mounted runtime stores CUDA libraries outside the reference layout |
| `TEXTURE_STEP1X_HEALTHCHECK_RUNTIME_IMPORTS` | readiness healthcheck `true` | Enables cached torch/CuPy/NVRTC runtime preflight when `healthcheck.py` checks `/health` |
| `TEXTURE_STEP1X_HEALTHCHECK_TIMEOUT` | `180s` | Docker healthcheck timeout; relevant to cold runtime imports only for readiness healthchecks |
| `TA_RENDER_PREVIEWS_ENABLED` | `false` | Enable material preview render step |
| `TA_RENDER_IMAGE_WIDTH` / `TA_RENDER_IMAGE_HEIGHT` | `1024` / `1024` | Final render dimensions |
| `TA_RENDER_PREVIEW_IMAGE_WIDTH` / `TA_RENDER_PREVIEW_IMAGE_HEIGHT` | `512` / `512` | Material preview render dimensions |
| `TA_SESSION_STORAGE_PATH` | `/var/texture-agent/sessions` | Session storage |
| `TA_SESSION_TTL_HOURS` | `24` | Session expiry |
| `TA_S3_ALLOWED_BUCKETS` | - | Exact bucket names allowed for client-supplied `s3_uri` inputs, separated by commas or whitespace; empty/unset rejects all S3 URI inputs |
| `TA_STORAGE_KIND` | `local` | Session store backend (`local` or `s3`) |
| `TA_STORAGE_S3_BUCKET` | `WU_S3_BUCKET` | S3 bucket for shared sessions |
| `TA_STORAGE_S3_PREFIX` | - | Prefix for shared session objects |
| `TA_STORAGE_S3_REGION` | `WU_S3_REGION` | S3 region |
| `TA_STORAGE_S3_PROFILE` | `WU_S3_PROFILE` | Optional AWS profile for local/dev runs |
| `TA_STORAGE_S3_ENDPOINT_URL` | - | Optional S3-compatible endpoint URL |
| `TA_STORAGE_S3_PRESIGN` | `true` | Return presigned artifact URLs when possible |
| `TA_STORAGE_S3_MAX_POOL_CONNECTIONS` | `64` | S3 client connection pool size |
| `TA_MAX_ACTIVE_SESSIONS` | `4` | Max concurrent pipelines, read when the registry starts. Invalid or negative values fall back to `4`; `0` permits no active executions. |
| `TA_CANCEL_DRAIN_TIMEOUT_SECONDS` | `30.0` | Seconds a cancelled request waits for a synchronous worker thread to stop before marking the session failed with a stalled-worker deletion guard |
| `TA_MAX_UPLOAD_SIZE_MB` | `500` | Max USD upload size |

`TA_S3_ALLOWED_BUCKETS` controls untrusted inbound S3 references and is
separate from `TA_STORAGE_S3_BUCKET`, which configures service-managed session
storage. Allowlist entries are exact bucket names only, not `s3://` URIs,
wildcards, or key prefixes; this release does not add application-level
key-prefix restrictions. A missing allowlist or a bucket that is not an exact
match returns the same generic HTTP `403` before any S3 request. Direct uploads
and existing-session inputs are unaffected.

Before upgrading a deployment whose clients use `s3_uri`, set the allowlist or
migrate those clients to direct uploads. Leaving it unset intentionally changes
all S3 URI requests to `403`; restart the service after changing the setting.
The Helm chart exposes the same setting as `textureAgent.s3AllowedBuckets`.

Use a dedicated intake bucket because every key in an allowed bucket is in
application scope. Restrict the service IAM role to only the required bucket
and prefixes as defense in depth; IAM scoping is not a substitute for the
allowlist.

For multi-instance deployments, use `TA_STORAGE_KIND=s3` and configure the S3
bucket, prefix, region, and credentials before increasing replicas. Local
storage is single-instance only because each pod sees only its own session
directory. The Helm chart exposes the same settings under `sessionStorage.*`;
keep `replicaCount: 1` unless `sessionStorage.kind` is `s3`. If both explicit
S3 credentials and `TA_STORAGE_S3_PROFILE` are configured, the explicit
credentials take precedence; if the named profile is unavailable, the service
falls back to the default boto3 credential chain.

## Project Structure

```
texture_agent_service/
├── client/                     # Python client (client.py)
├── docs/                       # Documentation (api.md REST reference)
├── service/                    # FastAPI app, routers, runtime, storage
├── tests/                      # Test suite
├── docker-compose.yml          # Docker Compose
├── Dockerfile                  # Service image
├── openapi.yaml                # API specification
└── pyproject.toml              # Install metadata
```
