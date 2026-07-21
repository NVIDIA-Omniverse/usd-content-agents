# Texture Gen Simple Service

A lightweight Texture Variation API service using the shared texture generation service harness and image generation models. Generates PBR texture sets (albedo, normal, ORM) from text prompts.

No GPU or conda required — runs anywhere the `world_understanding` package is installed.

## Usage

```bash
# From the repo root (with venv activated)
uvicorn apps.texture_gen_simple_service.app:app --port 8000

# Or directly
python apps/texture_gen_simple_service/app.py --port 8000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/texture-variations` | Submit a texture generation job |
| `GET` | `/v1/texture-variations/{job_id}` | Query job status |
| `DELETE` | `/v1/texture-variations/{job_id}` | Cancel a job |
| `GET` | `/health` | Health check |

The service uses the common `CreateJobRequest`, `JobStatus`, and
`HealthResponse` models from `apps.texture_gen_service_common`. Request parsing
therefore accepts the shared projection contract, including `target`,
`capabilities`, `reference_image_uris`, `turntable_video_uri`, and
`multiview_image_uris`. The simple backend currently generates from
`conditioning.text_prompt` and still returns full PBR maps.

`/health` reports readiness, whether the service is accepting jobs, active and
queued job counts, worker capacity, and backend capabilities. The common
response contract also permits degraded or albedo-only results from backends
that cannot produce normal or ORM maps, although this backend still attempts
albedo, normal, and ORM generation for every job.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEXTURE_OUTPUT_DIR` | System temp dir | Output directory for generated textures |
| `TEXTURE_GEN_BACKEND` | `nim` | Image generation backend (public default; routes to NVIDIA NIM at `build.nvidia.com`) |
| `TEXTURE_GEN_MODEL` | _(default)_ | Model override |
| `TEXTURE_GEN_BASE_URL` | _(unset)_ | Optional image-generation endpoint override, for example a local OpenAI-compatible FLUX NIM sidecar |
| `TEXTURE_GEN_API_KEY` | _(unset)_ | Optional endpoint-scoped API key. Leave unset for hosted providers so the backend can use its native key env (`NVIDIA_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY`, or `OPENAI_API_KEY`). Use `not-used` only for no-auth local endpoints |
| `TEXTURE_GEN_MAX_WORKERS` | `2` | Maximum concurrent texture generation jobs handled by the service worker pool |
| `TEXTURE_GEN_RETRY_ATTEMPTS` | `3` | Attempts for retryable image backend errors such as rate limits and transient 5xx responses |
| `TEXTURE_GEN_RETRY_BACKOFF_SEC` | `1.0` | Delay between retryable image backend attempts |
| `NVIDIA_API_KEY` | — | API key for the NVIDIA NIM backend |
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | — | API key for the Gemini backend |
| `OPENAI_API_KEY` | — | API key for the hosted OpenAI backend |

## Local FLUX NIM Sidecar

Run a local FLUX NIM sidecar, then start this service through the OpenAI-
compatible backend:

```bash
COLLECTION_IMAGE_GEN_PORT=8005 \
docker compose -f deploy/collection/docker-compose.image-gen.yml up -d

TEXTURE_GEN_BACKEND=openai \
TEXTURE_GEN_MODEL=black-forest-labs/flux.2-klein-4b \
TEXTURE_GEN_BASE_URL=http://localhost:8005/v1 \
TEXTURE_GEN_API_KEY=not-used \
uvicorn apps.texture_gen_simple_service.app:app --port 8000
```

## Example Request

```bash
curl -X POST http://localhost:8000/v1/texture-variations \
  -H "Content-Type: application/json" \
  -d '{
    "source_asset_uri": "file:///path/to/asset.usd",
    "conditioning": {
      "text_prompt": "heavily rusted metal with chipped paint",
      "multiview_image_uris": [
        "file:///path/to/front.png",
        "file:///path/to/side.png"
      ]
    },
    "configuration": {
      "strength": 0.9
    }
  }'
```
