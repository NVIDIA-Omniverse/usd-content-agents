# Material Agent Service API Reference

REST API for VLM-based material assignment to 3D USD files. The service accepts USD scene uploads, runs an async pipeline (optimize, render, build dataset, predict, apply, render final), and streams real-time progress via SSE.

**Base URL:** `http://localhost:8000`
**Interactive docs:** `GET /docs` (Swagger UI)
**OpenAPI:** [`../openapi.yaml`](../openapi.yaml)

---

## Table of Contents

- [Authentication](#authentication)
- [Root Endpoints](#root-endpoints)
- [Config](#config)
- [Pipeline](#pipeline)
- [Artifacts](#artifacts)
- [Assets](#assets)
- [Sessions](#sessions)
- [Materials](#materials)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Data Models](#data-models)
- [Configuration](#configuration)

---

## Authentication

No authentication is required by default. The service accepts all origins via permissive CORS.

---

## Root Endpoints

### `GET /health`

Health check. Includes backend credential readiness, image-generation readiness, and max active sessions.

**Response** `200`

```json
{
  "status": "healthy",
  "service": "Material Agent Service",
  "version": "0.5.2",
  "api_keys_configured": true,
  "image_gen_configured": true,
  "max_active_sessions": 3
}
```

The example shows the source fallback of `3`. The service image sets
`MA_MAX_ACTIVE_SESSIONS=8`, while local OVRTX Compose sets it to `1` unless an
operator overrides it. `/health` always reports the value enforced by the
process-wide registry.

---

## Config

### `GET /config/vlm-models`

List the VLM models exposed by the service. The list depends on which API keys are configured at startup.

**Response** `200` — list of `{backend, model, display_name}` entries.

---

## Pipeline

### `POST /pipeline/upload-usd`

Upload a USD file and create a new session. Use this when the USD lives on the client — the service stores it and returns a session you can then start a pipeline on.

**Request** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `usd_file` | file | USD asset to apply materials to. |

**Response** `201` — [`SessionCreated`](#sessioncreated)

### `POST /pipeline`

Start a material assignment pipeline on an existing session.

**Request body**

```json
{
  "session_id": "abc123",
  "materials_manifest": "default",
  "vlm": {
    "backend": "nim",
    "model": "google/gemma-4-31b-it"
  },
  "reference_images": ["ref1.jpg"],
  "generated_reference_id": "optional-generated-reference-id",
  "reference_pdfs": []
}
```

**Response** `202` — [`SessionCreated`](#sessioncreated)

#### Generated Material Library Mode

Generated material-library mode is opt-in per pipeline request. It asks the
pipeline VLM to describe asset-level material needs from reference imagery,
uses the deployment image-generation model to create texture maps, authors a
temporary USD material library, and then runs the normal material prediction and
apply pipeline with generated materials preferred over the default library.

Supported `multipart/form-data` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `enable_material_generation` | `false` | Enable generated material-library mode for this run. |
| `material_generation_guidance` | unset | Optional text guidance, for example "orange glossy enclosure, black rubber feet, brushed metal lid insert". |
| `material_generation_texture_size` | `1024` | Generated texture size in pixels. Accepted range: 64-4096. |

When enabled, the request must include at least one uploaded
`reference_images` entry or a `generated_reference_id` from
`POST /pipeline/{session_id}/generate-reference-image`. The service rejects the
request before queueing if `MA_IMAGE_GEN_*` deployment configuration is not
ready.

Generated material artifacts are written under
`generated_material_library/` in the session:

| Artifact | Description |
|----------|-------------|
| `materials.yaml` | Canonical manifest for the generated material library. |
| `material_library.usda` | USD layer containing the generated materials. |
| `material_generation_plan.yaml` | VLM-derived material plan used to drive texture generation. |
| texture maps | Generated albedo/roughness/normal or related maps referenced by the USD library. |

If prediction emits `__UNKNOWN__`, an empty material, or an invalid material
name that cannot be repaired, the pipeline binds the canonical
`__FALLBACK_MATERIAL__`: neutral mid-gray, matte, non-metallic, opaque plastic.
This fallback applies to default, custom, generated, regenerate, and refine
workflows.

#### Prim Clustering

Prim clustering is an opt-in optimization for large scenes. When
`enable_prim_clustering=true`, the service clusters visually similar rendered
prims before prediction, sends only cluster representatives to the VLM, then
expands representative predictions back to the original prims.

Use it for large scenes with repeated objects or repeated parts. Leave it off
for small scenes, scenes with few usable prim-only renders, or cases where each
small part needs a separate VLM decision.

Supported `multipart/form-data` fields:

| Field | Default | Description |
|-------|---------|-------------|
| `enable_prim_clustering` | `false` | Enable image-based prim clustering before prediction. |
| `cluster_min_prims` | `50` | Minimum prim count before clustering runs. |
| `cluster_embedding_backend` | `nim` | Embedding backend for NVIDIA image embeddings. |
| `cluster_embedding_model` | `nvidia/llama-nemotron-embed-vl-1b-v2` | NVIDIA VLM embedding model for hosted NIM and the local embedding sidecar. |
| `cluster_embedding_base_url` | unset | Optional embedding endpoint base URL. Request overrides are restricted to hosted NVIDIA URLs or the deployment-configured `MA_CLUSTER_EMBEDDING_BASE_URL`. |
| `cluster_embedding_max_workers` | `4` | Parallel embedding workers. |
| `cluster_embedding_batch_size` | `50` | Embedding batch size. |
| `cluster_max_size` | `25` | Maximum prims that can share one representative prediction before a cluster is split. |
| `cluster_similarity_threshold_low` | internal default | Optional similarity threshold for low-complexity prim clusters. |
| `cluster_similarity_threshold_medium` | internal default | Optional similarity threshold for medium-complexity prim clusters. |
| `cluster_similarity_threshold_high` | internal default | Optional similarity threshold for high-complexity prim clusters. |
| `cluster_report` | `true` | Generate a clustering HTML report when clustering runs. |

Invalid combinations fail before background execution. For example,
`enable_prim_clustering=true` requires a prediction or benchmark step so the
representative predictions can be expanded.

### `POST /pipeline/{session_id}/generate-reference-image`

Generate an AI reference image from the uploaded input preview and a text prompt.
The response includes a `reference_id`; pass it as `generated_reference_id` to
`POST /pipeline` to use that generated image for the run.

**Response** `200`

```json
{
  "status": "ok",
  "reference_id": "generated-reference-id",
  "image_url": "/assets/session-id/generated-ref/generated-reference-id"
}
```

### `GET /pipeline/{session_id}/status`

Get the current pipeline status — step progress, active step, completion fraction.

**Response** `200` — [`PipelineStatus`](#pipelinestatus)

### `GET /pipeline/{session_id}/results`

Get the final pipeline results. Returns `409 Conflict` until the pipeline completes.

**Response** `200` — [`PipelineResults`](#pipelineresults)

### `POST /pipeline/{session_id}/cancel`

Request cancellation of a running pipeline.

**Response** `200`

### `GET /pipeline/{session_id}/events`

Subscribe to real-time pipeline events via Server-Sent Events. See [Server-Sent Events](#server-sent-events-sse) below.

**Response** `200` — `text/event-stream`

### `GET /pipeline/{session_id}/event-log`

Get the persisted event history for a session. Use this to replay missed
progress updates after disconnects or after a session reaches a terminal state;
use `/events` for live SSE streaming.

**Response** `200`

Example payload: `apps/material_agent_service/examples/event_log_response.json`

### `POST /pipeline/{session_id}/regenerate`

Re-run selected pipeline steps from cached session data. This is useful for
trying a new prompt or re-running `apply`/`render` without uploading the USD
again.

**Request** `application/json`

Example request body: `apps/material_agent_service/examples/regenerate_request.json`

`steps` accepts material pipeline step names such as
`build_dataset_prepare_dataset`, `build_dataset_usd`, `predict`, `apply`, and
`render`. When the original session used prim clustering, regenerating
prediction-related steps also restores the clustering expansion steps needed
for a consistent result. `user_prompt` overrides the prompt for the regenerated
run, and `layer_only=true` makes the apply step emit only a material binding
layer.

**Response** `202` — [`SessionCreated`](#sessioncreated) (see the endpoint
usage flow in `apps/material_agent_service/examples/regenerate_client_usage.py`)

**Errors:** `400` while the pipeline is pending/running/cancelling or when the
session has no cached input USD; `404` when the session does not exist.

---

## Artifacts

All artifact endpoints require the corresponding pipeline step to have completed.

### `GET /artifacts/{session_id}/output`

Download the output USD file with materials applied.

**Response** `200` — `application/octet-stream`

### `GET /artifacts/{session_id}/final-render`

Download the final render image of the materialized asset.

**Response** `200` — `image/png`

Large-scene public quickstart assets are shipped with the service under
`apps/material_agent_service/examples/large_scene/README.md`.

### `GET /artifacts/{session_id}/predictions`

Download the predictions JSONL file (one prediction per line).

**Response** `200` — `application/x-ndjson`

### `GET /artifacts/{session_id}/scene-manifest`

Download the large-scene `manifest.json` produced by scene analysis and
per-asset execution.

**Response** `200` — `application/json`

### `GET /artifacts/{session_id}/scene-validation-report`

Download the large-scene validation report JSON.

**Response** `200` — `application/json`

### `GET /artifacts/{session_id}/scene-predictions`

Download collated large-scene predictions JSONL. Each line includes the
source sub-asset or payload metadata plus the original prediction record.

**Response** `200` — `application/x-ndjson`

### `GET /artifacts/{session_id}/cluster-map`

Download the prim clustering map JSONL file. Each row maps an original prim to
its cluster ID, representative prim, cluster size, and complexity tier.

**Response** `200` — `application/x-ndjson`

### `GET /artifacts/{session_id}/cluster-report`

View the bounded prim clustering HTML report.

**Response** `200` — `text/html`

### `GET /artifacts/{session_id}/cluster-summary`

Download a lightweight JSON summary of cluster counts, reduction, tier
breakdown, safety-cap behavior, and report limits.

**Response** `200` — `application/json`

### `GET /artifacts/{session_id}/cluster-representatives`

Download the representative-only dataset used for clustered prediction.

**Response** `200` — `application/x-ndjson`

### `GET /artifacts/{session_id}/report`

View the prediction HTML report (rendered side-by-side of predictions and ground truth, if available).

**Response** `200` — `text/html`

### `GET /artifacts/{session_id}/optimization-report`

View the USD optimization JSON report (output of the `optimize_usd` step).

**Response** `200` — `application/json`

---

## Assets

Input assets are distinguished from output artifacts — assets are what the user uploaded, artifacts are what the pipeline produced.

### `GET /assets/{session_id}/input-render`

Get a render of the input USD (before any material assignment).

### `GET /assets/{session_id}/previews`

List all preview images rendered during the pipeline.

**Response** `200` — list of preview filenames

### `GET /assets/{session_id}/preview/{image_name}`

Download a specific preview image.

### `GET /assets/{session_id}/references`

List reference images for the session.

### `GET /assets/{session_id}/reference/{image_name}`

Download a specific reference image.

### `GET /assets/{session_id}/reference-pdfs`

List reference PDFs for the session.

### `GET /assets/{session_id}/reference-pdf/{pdf_name}`

Download a specific reference PDF.

---

## Sessions

### `GET /sessions`

List all sessions on the server.

**Response** `200` — `list[SessionSummary]`

### `GET /sessions/usage`

Get aggregate usage statistics (counts by status, total compute time, etc.).

### `GET /sessions/{session_id}`

Get full session details including pipeline configuration and current status.

### `DELETE /sessions/{session_id}`

Delete a session and all its artifacts.

**Response** `204`

### `POST /sessions/admin/cleanup`

Trigger manual cleanup of expired sessions (normally runs on a schedule).

---

## Materials

### `GET /materials`

List available materials from the default material library (or the one configured via `MA_MATERIAL_LIBRARY`).

**Response** `200` — list of `{name, description, binding, icon_url, icon_path}` entries.
`icon_url` / `icon_path` are `null` when the library does not ship thumbnails.

### `GET /materials/icon/{material_name}`

Get the icon image (thumbnail) for a named material.

**Response** `200` — `image/png`

### `GET /materials/template`

Download the default materials template ZIP. Use this as a starting point for building your own material library.

The bundled default template ships `materials.yaml` plus the USD library.
Thumbnail icons are optional and are not included by default.

**Response** `200` — `application/zip`

---

## Server-Sent Events (SSE)

### `GET /pipeline/{session_id}/events`

Subscribe to real-time pipeline events. Events include step start/end, progress updates, and errors.

**Response** `200` — `text/event-stream`

Example events:

```
event: step_started
data: {"step": "build_dataset_usd", "started_at": "..."}

event: step_progress
data: {"step": "predict", "current": 7, "total": 20}

event: step_completed
data: {"step": "apply", "ended_at": "...", "success": true}

event: pipeline_completed
data: {"status": "completed"}
```

Clients should reconnect on disconnect. Use
`GET /pipeline/{session_id}/event-log` to replay persisted event history.

---

## Data Models

### `SessionCreated`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier. |
| `status` | string | Initial status (usually `queued`). |
| `created_at` | datetime | ISO timestamp. |

### `PipelineStatus`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID. |
| `status` | `queued \| running \| completed \| failed \| cancelled` | Overall pipeline state. |
| `current_step` | `CurrentStepInfo \| null` | The step currently executing (if any). |
| `completed_steps` | `list[CompletedStepInfo]` | Steps that have finished. |
| `progress` | `OverallProgress` | Fraction of total steps completed. |

### `PipelineResults`

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session ID. |
| `output_usd_url` | string | Path to the materialized USD (served by `/artifacts/.../output`). |
| `final_render_url` | string | Path to the final render image. |
| `predictions_url` | string | Path to the predictions JSONL. |
| `report_url` | string | Path to the HTML report. |
| `metrics` | object | Aggregate metrics (material coverage, prediction confidence, etc.). |
| `stats.cluster_prims_ran` | boolean | Whether prim clustering ran for this pipeline. |
| `stats.cluster_representative_count` | integer | Number of representative prims sent to prediction. |
| `stats.cluster_reduction_percent` | number | Approximate VLM prediction reduction from clustering. |
| `stats.cluster_max_size` | integer/null | Effective max prims per propagated representative prediction. |
| `stats.cluster_capped_count` | integer | Number of oversized visual clusters split by the max-size cap. |

### Pipeline Steps

The material-agent pipeline runs the following steps (some opt-in):

1. `validate_input` — Sanity-check the USD and configuration
2. `optimize_usd` — Flatten and deinstance via scene optimizer
3. `render_preview` — Lightweight whole-scene preview rendering
4. `generate_reference_image` — Generate photorealistic reference images (opt-in)
5. `generate_material_library` — Generate an asset-specific material library from reference images (opt-in)
6. `build_dataset_usd` — Render prim views for VLM input
7. `build_dataset_prepare_dataset` — Prepare dataset entries with material specs
8. `cluster_prims` — Cluster visually similar prims before prediction (opt-in)
9. `predict` — VLM inference for material assignment
10. `expand_cluster_predictions` — Expand representative predictions to cluster members (auto-enabled when needed)
11. `validate_predictions` — Validate/repair predicted material names against the library
12. `harmonize_predictions` — Resolve conflicts for instanced parts
13. `apply` — Apply predicted materials to USD
14. `restore_usd` — Restore the original USD hierarchy (reverses optimize_usd changes)
15. `validate_output` — Sanity-check the output USD
16. `render` — Final render of the materialized asset

---

## Configuration

The service reads its configuration from environment variables at startup. See `.env_example` for the full list. Key settings:

| Variable | Description |
|----------|-------------|
| `NVIDIA_API_KEY` | Required if using `nim` VLM backend |
| `OPENAI_API_KEY` | Required if using `openai` backend |
| `ANTHROPIC_API_KEY` | Required if using `anthropic` backend |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Required if using `gemini` backend |
| `MA_SESSION_STORAGE_PATH` | Where session directories are written |
| `MA_MAX_UPLOAD_SIZE_MB` | Max USD file size for `/pipeline/upload-usd` |
| `MA_MAX_ACTIVE_SESSIONS` | Max concurrent pipelines, read when the registry starts. The source fallback is `3`; invalid or negative values fall back to `3`; `0` permits no active executions. The service image sets `8`; local OVRTX Compose sets `1`. |
| `MA_MAX_RENDER_NUM_WORKERS` | Max accepted `render_num_workers` override; local OVRTX compose defaults to `1` |
| `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` | Process-wide render request cap; local OVRTX compose defaults to `1` |
| `MA_VLM_BACKEND`, `MA_VLM_MODEL` | Default VLM backend + model |
| `MA_LLM_BACKEND`, `MA_LLM_MODEL` | Default LLM backend + model for validate/harmonize |
| `MA_IMAGE_GEN_BACKEND` | Image generation backend for generated reference images and generated material-library textures (default `gemini`) |
| `MA_IMAGE_GEN_MODEL` | Optional shared image-generation model override |
| `MA_IMAGE_GEN_BASE_URL` | Optional shared image-generation API base URL override |
| `MA_IMAGE_GEN_API_KEY` | Optional shared image-generation API key; use `not-used` only for explicit no-auth local endpoints |
| `MA_CLUSTER_EMBEDDING_BACKEND` | Default embedding backend for opt-in prim clustering (default `nim`) |
| `MA_CLUSTER_EMBEDDING_MODEL` | Default embedding model for opt-in prim clustering |
| `MA_CLUSTER_EMBEDDING_BASE_URL` | Optional clustering embedding API base URL override |
| `MA_CLUSTER_EMBEDDING_API_KEY` | Optional endpoint-specific clustering embedding key |
| `MA_CLUSTER_EMBEDDING_MAX_WORKERS` | Default clustering embedding worker count |
| `MA_CLUSTER_EMBEDDING_BATCH_SIZE` | Default clustering embedding batch size |
| `MA_CLUSTER_MIN_PRIMS` | Default minimum prim count before clustering runs |
| `MA_CLUSTER_MAX_SIZE` | Default max prims per propagated representative prediction |
| `MA_RENDERER_BACKEND` | `ovrtx` or `remote` |

Prediction concurrency is request-scoped through the `vlm_max_workers` form
field. Render concurrency is controlled separately by `render_num_workers` and
the process-wide `WU_NVCF_GLOBAL_MAX_CONCURRENT_REQUESTS` cap.

Troubleshooting:

- Hosted `nim` embeddings require a valid `NVIDIA_API_KEY`, unless
  `MA_CLUSTER_EMBEDDING_BASE_URL` points at an explicit local no-auth endpoint
  and `MA_CLUSTER_EMBEDDING_API_KEY=not-used`.
- If cluster stats show no reduction, the scene may be below
  `cluster_min_prims`, lack prim-only render images, or contain visually
  distinct prims.
- If a repeated-object scene still over-propagates predictions, lower
  `cluster_max_size` for that run. Oversized clusters are split into stable
  chunks before representative selection.
- Cluster reports are bounded by default so the HTML remains usable on large
  scenes. Use `cluster-summary`, `cluster-map`, and `cluster-representatives`
  for complete machine-readable data, or set `cluster_report=false` per
  request to skip HTML generation.
