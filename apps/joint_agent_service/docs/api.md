# Joint Agent Service API Reference

REST API for VLM-based asset classification of USD files. The service accepts USD scene uploads, runs an async pipeline (optimize, render, build dataset, predict), and streams real-time progress via SSE.

**Base URL:** `http://localhost:8000`
**Interactive docs:** `GET /docs` (Swagger UI)

---

## Table of Contents

- [Authentication](#authentication)
- [Root Endpoints](#root-endpoints)
- [Pipeline](#pipeline)
- [Artifacts](#artifacts)
- [Sessions](#sessions)
- [Server-Sent Events (SSE)](#server-sent-events-sse)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Configuration](#configuration)

---

## Authentication

No authentication is required. The service accepts all origins via permissive CORS.

Optional: set `JOINT_AGENT_TOKEN` and pass it as `Authorization: Bearer <token>` from clients. The service does not currently enforce this.

---

## Root Endpoints

### `GET /`

Redirects to `/api`.

### `GET /api`

Returns service info and a map of all available endpoints.

**Response** `200`
```json
{
  "service": "Joint Agent Service",
  "version": "0.5.2",
  "docs": "/docs",
  "health": "/health",
  "api": {
    "pipeline": {
      "create": "POST /pipeline",
      "status": "GET /pipeline/{session_id}/status",
      "results": "GET /pipeline/{session_id}/results",
      "cancel": "POST /pipeline/{session_id}/cancel?run_id={run_id}",
      "events": "GET /pipeline/{session_id}/events",
      "regenerate": "POST /pipeline/{session_id}/regenerate"
    },
    "artifacts": {
      "predictions": "GET /artifacts/{session_id}/predictions",
      "report": "GET /artifacts/{session_id}/report",
      "dataset": "GET /artifacts/{session_id}/dataset",
      "joint_rigger_output": "GET /artifacts/{session_id}/joint-rigger-output",
      "joint_rigger_output_filename": "rigged.usdz (preferred; legacy responses may use rigged.usd)"
    },
    "sessions": {
      "list": "GET /sessions",
      "get": "GET /sessions/{session_id}",
      "delete": "DELETE /sessions/{session_id}"
    }
  }
}
```

### `GET /health`

Health check.

**Response** `200`
```json
{
  "status": "healthy",
  "service": "Joint Agent Service",
  "version": "0.5.2",
  "api_keys_configured": true,
  "max_active_sessions": 1,
  "capabilities": {
    "joint_rigger": {
      "owned_core_available": true,
      "usd_joint_rigger_available": false,
      "import_error_type": "ModuleNotFoundError"
    }
  }
}
```

`owned_core_import_error_type` is present only when the built-in core cannot be
imported. The legacy `import_error_type` field describes the optional external
`usd_joint_rigger` package.

---

## Pipeline

### Upload USD

```
POST /pipeline/upload-usd
```

Upload a USD file or import one from an authorized S3 bucket and create a
session without starting the pipeline. Use the returned `session_id` with
`POST /pipeline` to start processing later.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usd_file` | file | conditional | USD file (`.usd`, `.usda`, `.usdc`, `.usdz`). Required when `s3_uri` is absent. |
| `s3_uri` | string | conditional | S3 URI to a USD file in a bucket allowed by `JA_S3_ALLOWED_BUCKETS`. Required when `usd_file` is absent. |

**Response** `201`
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "ready",
  "message": "USD uploaded successfully",
  "estimated_duration_minutes": 0
}
```

**Errors:**
- `400` Neither or both of `usd_file` and `s3_uri` provided; invalid file extension
- `403` S3 URI is not permitted by the configured bucket allowlist, or S3 access is denied
- `413` File exceeds an operator-configured positive `JA_MAX_UPLOAD_SIZE_MB`

---

### Create Pipeline

```
POST /pipeline
```

Create and execute an asset classification pipeline. Supports three modes:
1. **New upload:** provide `usd_file` to create a new session and start processing.
2. **Existing session:** provide `session_id` (from `/pipeline/upload-usd`) to start processing a previously uploaded file.
3. **S3 source:** provide `s3_uri` for a USD in a bucket allowed by `JA_S3_ALLOWED_BUCKETS`.

Pipeline execution is async -- the endpoint returns `202` immediately.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `usd_file` | file | conditional | USD file. Required if neither `session_id` nor `s3_uri` is provided. |
| `session_id` | string | conditional | Existing session ID. Required if neither `usd_file` nor `s3_uri` is provided. |
| `s3_uri` | string | conditional | S3 URI to a USD file in a bucket allowed by `JA_S3_ALLOWED_BUCKETS`. Required if neither `usd_file` nor `session_id` is provided. |
| `user_prompt` | string | no | Custom prompt for the VLM prediction step. |
| `render_backend` | string | no | Rendering backend: `remote` (default, through `RENDER_ENDPOINT`), `warp` (local CUDA), `ovrtx` (local Vulkan), or `mock` (deterministic CPU-only test images). Omission uses the configured service default. |
| `apply_joint_rigger` | boolean | no | Enable the opt-in Research Preview Joint Rigger. Defaults to `false`. |
| `joint_rigger_adapter` | string | no | `owned_core`, `mock`, or `usd_joint_rigger`. Omission selects `owned_core` when the apply step is enabled. |
| `joint_rigger_apply_masses` | boolean | no | Must be `false` for topology-only `owned_core`; `true` is rejected. |
| `joint_rigger_apply_collision` | boolean | no | Must be `false` for topology-only `owned_core`; `true` is rejected. |

**Response** `202`
```json
{
  "session_id": "a1b2c3d4-...",
  "run_id": "0123456789abcdef0123456789abcdef",
  "status": "pending",
  "message": "Pipeline queued for execution",
  "estimated_duration_minutes": 15
}
```

**Errors:**
- `400` No input source provided; invalid file extension; input USD not found for session
- `403` S3 URI is not permitted by the configured bucket allowlist, or S3 access is denied
- `404` Session not found (when using `session_id`)
- `413` File too large

---

### Get Pipeline Status

```
GET /pipeline/{session_id}/status
```

Real-time pipeline status with step-level progress. Reads from the in-memory event bus for active sessions, falls back to disk for completed ones.

**Response** `200` -- [PipelineStatus](#pipelinestatus)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "running",
  "current_step": {
    "name": "predict",
    "display_name": "Running VLM Predictions",
    "started_at": "2026-02-24T10:05:00Z",
    "progress": {
      "current": 47,
      "total": 95,
      "percent": 49,
      "message": "Predicted /World/Part_47"
    },
    "elapsed_seconds": 120
  },
  "completed_steps": [
    {
      "name": "optimize_usd",
      "display_name": "Optimizing USD",
      "started_at": "2026-02-24T10:00:00Z",
      "completed_at": "2026-02-24T10:01:30Z",
      "duration_seconds": 90,
      "stats": {}
    }
  ],
  "overall_progress": {
    "current_step": 5,
    "total_steps": 7,
    "percent": 78,
    "estimated_remaining_seconds": 180
  },
  "preview_images": ["/artifacts/a1b2c3d4-.../preview/abc123.png"],
  "can_cancel": true,
  "elapsed_seconds": 300,
  "created_at": "2026-02-24T10:00:00Z",
  "updated_at": "2026-02-24T10:05:00Z"
}
```

**Errors:**
- `404` Session not found

---

### Get Pipeline Results

```
GET /pipeline/{session_id}/results
```

Returns final results when the pipeline has completed, or error details if it failed.

**Response** `200` (completed) -- [PipelineResults](#pipelineresults)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "completed",
  "stats": {
    "prims_processed": 142,
    "images_generated": 284,
    "predictions_made": 142,
    "articulation_candidates": 12
  },
  "download_urls": {
    "predictions": "/artifacts/a1b2c3d4-.../predictions",
    "articulation_candidates": "/artifacts/a1b2c3d4-.../articulation-candidates",
    "articulation_report": "/artifacts/a1b2c3d4-.../articulation-report",
    "report": "/artifacts/a1b2c3d4-.../report",
    "dataset": "/artifacts/a1b2c3d4-.../dataset"
  },
  "duration_seconds": 600,
  "completed_at": "2026-02-24T10:10:00Z"
}
```

**Response** `200` (failed) -- [PipelineError](#pipelineerror)
```json
{
  "session_id": "a1b2c3d4-...",
  "status": "failed",
  "error_message": "VLM inference timeout",
  "failed_step": "predict",
  "completed_steps": ["optimize_usd", "build_dataset_usd"],
  "partial_results": null
}
```

**Errors:**
- `202` Pipeline still running (check `/status` for progress)
- `404` Session not found

---

### Cancel Pipeline

```
POST /pipeline/{session_id}/cancel?run_id={run_id}
```

Cancel the exact running or pending pipeline generation returned by create or
regenerate. A delayed cancellation for an older generation is rejected.

**Response** `200`
```json
{
  "session_id": "a1b2c3d4-...",
  "run_id": "0123456789abcdef0123456789abcdef",
  "status": "cancelling",
  "message": "Pipeline cancellation requested"
}
```

**Errors:**
- `400` Pipeline already completed/failed/cancelled
- `409` The active run changed before cancellation could be accepted
- `422` Missing or malformed run-generation token
- `404` Session not found

---

### Stream Events (SSE)

```
GET /pipeline/{session_id}/events
```

Server-Sent Events stream for real-time progress. See [SSE section](#server-sent-events-sse) for details.

**Errors:**
- `404` Session not found

---

### Regenerate Pipeline

```
POST /pipeline/{session_id}/regenerate
```

Re-run specific pipeline steps using cached data from a previous run. Useful for re-running the `predict` step with a different prompt without re-rendering.
Pipeline configurations, dataset inputs, and prediction prerequisites are restored
from immutable, generation-bound snapshots selected by session metadata.
Superseded snapshots remain until session TTL cleanup or deletion; automatic
per-rerun pruning is post-0.5 work.

**Request:** `application/json` -- [RegenerateRequest](#regeneraterequest)
```json
{
  "steps": ["predict"],
  "user_prompt": "Classify each material as metal, plastic, or fabric"
}
```

**Response** `202` -- [PipelineRunCreated](#pipelineruncreated)
```json
{
  "session_id": "a1b2c3d4-...",
  "run_id": "fedcba9876543210fedcba9876543210",
  "status": "pending",
  "message": "Regenerating steps: predict"
}
```

**Errors:**
- `409` Another run owns the session, or required generation-bound cache data is unavailable
- `400` Original config not found or invalid regeneration request
- `404` Session not found

---

### Get Event Log

```
GET /pipeline/{session_id}/event-log
```

Get the full persisted event history for a session. Useful for replaying progress after a session completes.

**Response** `200`
```json
{
  "events": [
    {
      "session_id": "a1b2c3d4-...",
      "step": "optimize_usd",
      "state": "running",
      "percent": 0,
      "message": "Starting USD optimization",
      "timestamp": "2026-02-24T10:00:00Z"
    }
  ],
  "total": 42
}
```

**Errors:**
- `404` Session not found

---

## Artifacts

### Download Predictions

```
GET /artifacts/{session_id}/predictions
```

Download the predictions file (JSONL). For the default 0.5 pipeline this file
includes the post-predict `consistency_pass` annotations.

**Response** `200`
Content-Type: `application/x-ndjson`
Filename: `predictions.jsonl`

**Errors:**
- `404` Session or predictions not found

---

### Download Articulation Candidates

```http
GET /artifacts/{session_id}/articulation-candidates
```

Download the Stage 2 articulation candidates JSON report. This is a review
artifact; it does not apply USD joints.

**Response** `200`
Content-Type: `application/json`
Filename: `articulation_candidates.json`

**Errors:**
- `404` Session or articulation candidates not found

---

### Download Joint Rigger Output

```http
GET /artifacts/{session_id}/joint-rigger-output
```

New `owned_core` runs return the self-contained `rigged.usdz` package. Completed
results bind `joint_rigger_artifact_keys` to immutable keys under the exact
current-run publication directory, so a reused session cannot select or
overwrite an artifact from another run. The endpoint uses USDZ-first probing
only for legacy completed sessions without the binding map. A non-empty
publication also includes an opaque `joint_rigger_publication_id`. Downloads
open a stable byte snapshot, then revalidate that publication identity before
responding. A result URL is advertised only when the current run published the
corresponding artifact.

Both contract-derived paths author accepted joint topology and source-backed
limits. For aggregate or multi-root contracts, V2 additionally authors exact
aggregate rigid-link membership and articulation roots; ordinary one-root
existing-link contracts retain V1. The owned path does not author rigid bodies,
masses, colliders, drives, joint state, or mimic schemas, and does not prove
simulation readiness.
Service-generated candidates are limited to revolute and prismatic joints in
0.5. Empty, all-unready, or policy-blocked candidate inputs return diagnostics
without publishing a generated package.

---

### View Articulation Report

```http
GET /artifacts/{session_id}/articulation-report
```

View the Stage 2 articulation candidate HTML report in the browser.

**Response** `200`
Content-Type: `text/html`

**Errors:**
- `404` Session or articulation report not found

---

### View Report

```
GET /artifacts/{session_id}/report
```

View the HTML prediction report in the browser. The report is generated on-demand if it doesn't already exist.

**Response** `200`
Content-Type: `text/html`

**Errors:**
- `404` Session not found; predictions not available yet; dataset not available
- `500` Report generation failed

---

### Download Dataset

```
GET /artifacts/{session_id}/dataset
```

Download the dataset file (JSONL).

**Response** `200`
Content-Type: `application/x-ndjson`
Filename: `dataset.jsonl`

**Errors:**
- `404` Session or dataset not found

---

## Sessions

### List Sessions

```
GET /sessions
```

List all sessions sorted by creation time (newest first).

**Response** `200`
```json
{
  "sessions": [
    {
      "session_id": "a1b2c3d4-...",
      "status": "completed",
      "created_at": "2026-02-24T10:00:00Z",
      "updated_at": "2026-02-24T10:10:00Z",
      "elapsed_seconds": 600,
      "config": {
        "project_name": "my_scene",
        "usd_path": "/var/joint-agent/sessions/a1b2c3d4-.../input/scene.usd",
        "has_usd_upload": true,
        "user_prompt": null
      }
    }
  ],
  "total": 1
}
```

---

### Get Session

```
GET /sessions/{session_id}
```

Get full session metadata.

**Response** `200` -- Full `session.json` contents (see [Session Metadata](#session-metadata)).

**Errors:**
- `404` Session not found

---

### Delete Session

```
DELETE /sessions/{session_id}
```

Delete a session and all its artifacts. Cancels any running pipeline first.

**Response** `204` No Content

**Errors:**
- `404` Session not found
- `500` Deletion failed after retries

---

## Server-Sent Events (SSE)

Connect to `GET /pipeline/{session_id}/events` to receive real-time progress updates.

### Event Types

| Event | Description |
|-------|-------------|
| `progress` | Pipeline step progress update ([ProgressEvent](#progressevent)) |
| `ping` | Keepalive sent every 30 seconds |
| `done` | Pipeline completed, failed, or cancelled -- stream closes after this |

### JavaScript Example

```javascript
const events = new EventSource("/pipeline/a1b2c3d4-.../events");

events.addEventListener("progress", (e) => {
  const data = JSON.parse(e.data);
  console.log(`[${data.step}] ${data.state} ${data.percent}%: ${data.message}`);
});

events.addEventListener("done", (e) => {
  const data = JSON.parse(e.data);
  console.log(`Pipeline ${data.final_state}`);
  events.close();
});
```

### ProgressEvent Payload

```json
{
  "session_id": "a1b2c3d4-...",
  "step": "predict",
  "state": "running",
  "current": 47,
  "total": 95,
  "percent": 49,
  "message": "Predicted /World/Part_47",
  "timestamp": "2026-02-24T10:05:00Z",
  "extra": { "prim_id": "/World/Part_47" },
  "overall_percent": 62
}
```

---

## Data Models

### SessionCreated

Returned when a session is created or a pipeline is queued.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID session identifier |
| `status` | string | `"pending"`, `"ready"` |
| `message` | string | Human-readable message |
| `estimated_duration_minutes` | int or null | Rough time estimate |

### PipelineRunCreated

Returned when a pipeline or regeneration run is accepted. It includes all
`SessionCreated` fields plus the generation token used for cancellation.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | 32-character token identifying the exact accepted run |

### PipelineStatus

Detailed pipeline execution status.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled`, `cancelling` |
| `current_step` | [CurrentStepInfo](#currentstepinfo) or null | Currently executing step |
| `completed_steps` | [CompletedStepInfo](#completedstepinfo)[] | Steps that have finished |
| `overall_progress` | [OverallProgress](#overallprogress) | Aggregate progress |
| `preview_images` | string[] | URLs to rendered preview thumbnails |
| `can_cancel` | bool | Whether the pipeline can be cancelled |
| `elapsed_seconds` | int | Total elapsed time |
| `created_at` | string | ISO 8601 timestamp |
| `updated_at` | string | ISO 8601 timestamp |

### CurrentStepInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Internal step name |
| `display_name` | string | Human-readable name |
| `started_at` | string | ISO 8601 timestamp |
| `progress` | [StepProgress](#stepprogress) | Step-level progress |
| `elapsed_seconds` | int | Time since step started |

### CompletedStepInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Internal step name |
| `display_name` | string | Human-readable name |
| `started_at` | string | ISO 8601 timestamp |
| `completed_at` | string | ISO 8601 timestamp |
| `duration_seconds` | int | Step duration |
| `stats` | object | Step-specific statistics |

### StepProgress

| Field | Type | Description |
|-------|------|-------------|
| `current` | int | Items processed so far |
| `total` | int | Total items to process |
| `percent` | int | Percentage complete (0-100) |
| `message` | string | Human-readable progress message |

### OverallProgress

| Field | Type | Description |
|-------|------|-------------|
| `current_step` | int | Current step number (1-indexed) |
| `total_steps` | int | Total pipeline steps |
| `percent` | int | Overall percentage (0-100) |
| `estimated_remaining_seconds` | int or null | Estimated time remaining |

### PipelineResults

Returned when the pipeline completes successfully.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `"completed"` |
| `stats` | object | Counts plus optional Joint Rigger status, boolean artifact flags, exact current-run `joint_rigger_artifact_keys`, and an opaque `joint_rigger_publication_id` |
| `download_urls` | object | `{predictions, articulation_candidates, articulation_report, report, dataset}` -- relative URL paths |
| `duration_seconds` | int | Total pipeline duration |
| `completed_at` | string | ISO 8601 timestamp |

### PipelineError

Returned when the pipeline fails.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `status` | string | `"failed"` |
| `error_message` | string | Error description |
| `failed_step` | string | Step that failed |
| `completed_steps` | string[] | Steps completed before failure |
| `partial_results` | object or null | Any partial results available |

### RegenerateRequest

Request body for regenerating specific pipeline steps.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `steps` | [PipelineStep](#pipelinestep)[] | yes | Steps to re-run |
| `user_prompt` | string or null | no | Override prompt for the prediction step |

### PipelineStep

Enum of available pipeline steps:

| Value | Description |
|-------|-------------|
| `optimize_usd` | Optimize USD scene structure |
| `identify_asset` | Identify the overall asset |
| `analyze_structure` | Analyze hierarchy and structure |
| `build_dataset_usd` | Render images from USD prims |
| `build_dataset_prepare_dataset` | Prepare dataset with prompts |
| `predict` | Run VLM predictions |
| `consistency_pass` | Annotate repeated-part prediction consistency |
| `infer_articulation_candidates` | Infer report-only Stage 2 articulation candidates |
| `restore_usd` | Restore predictions to original USD paths |

### ProgressEvent

SSE event payload for real-time progress.

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Session identifier |
| `step` | string | Step name |
| `state` | string | `queued`, `running`, `completed`, `failed`, `cancelled` |
| `current` | int or null | Items processed |
| `total` | int or null | Total items |
| `percent` | int or null | Step percentage (0-100) |
| `message` | string or null | Progress message |
| `timestamp` | string | ISO 8601 timestamp |
| `extra` | object or null | Step-specific data |
| `overall_percent` | int or null | Pipeline-wide percentage (0-100) |

### Session Metadata

Full session metadata stored on disk (`session.json`).

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | UUID identifier |
| `created_at` | string | ISO 8601 creation time |
| `updated_at` | string | ISO 8601 last update time |
| `status` | string | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `current_step` | object or null | Current step info |
| `completed_steps` | object[] | Completed steps |
| `overall_progress` | object | Aggregate progress |
| `preview_images` | string[] | Preview image filenames |
| `can_cancel` | bool | Cancellation availability |
| `elapsed_seconds` | int | Elapsed time |
| `config` | object | `{project_name, usd_path, has_usd_upload, user_prompt}` |
| `ttl_expires_at` | string | Expiration timestamp |
| `results` | object | Final stats |
| `duration_seconds` | int | Total duration |
| `completed_at` | string | Completion timestamp |
| `timings` | object | Per-step durations |

---

## Error Handling

All errors return JSON with a `detail` field:

```json
{
  "detail": "Session not found"
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `201` | Resource created (upload-usd) |
| `202` | Accepted -- pipeline queued or still running |
| `204` | Deleted successfully (no body) |
| `400` | Bad request (missing params, invalid state) |
| `403` | Client-supplied S3 URI is outside the configured bucket allowlist, or S3 access is denied |
| `404` | Session or artifact not found |
| `413` | File too large |
| `500` | Internal server error |

---

## Configuration

All settings use the `JA_` environment variable prefix.

### Service Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `JA_SESSION_STORAGE_PATH` | `/var/joint-agent/sessions` | Session storage directory (falls back to `./sessions` in dev) |
| `JA_SESSION_TTL_HOURS` | `24` | Hours before sessions are auto-cleaned |
| `JA_RUN_CLAIM_LEASE_SECONDS` | `300` | Cross-instance run lease; a crashed owner can be replaced after this interval |
| `JA_RUN_CLAIM_HEARTBEAT_SECONDS` | `60` | Run lease renewal interval; must be shorter than the lease |
| `JA_MAX_ACTIVE_SESSIONS` | `1` | Max concurrent pipeline executions, read when the registry starts. Invalid or negative values fall back to `1`; `0` permits no active executions. |
| `JA_MAX_UPLOAD_SIZE_MB` | `0` | Optional upload limit in MiB; `0` disables the limit. Uploads stream to a temporary snapshot before atomic publication. |
| `JA_S3_ALLOWED_BUCKETS` | empty | Comma- or whitespace-separated exact bucket names allowed for client-supplied `s3_uri` inputs. Empty rejects all such inputs. |

### Client-supplied S3 authorization

`JA_S3_ALLOWED_BUCKETS` is deliberately fail-closed. When it is unset or empty,
or when the requested bucket is not an exact match, the service rejects the
request before any S3 operation with HTTP `403` and the generic detail
`S3 URI is not permitted by the service's configured bucket allowlist`. The
same response is used in both cases and does not reveal whether an object
exists.

Upgrade deployments that already accept `s3_uri` by setting this variable
before rollout, or migrate callers to direct uploads. Otherwise all S3 URI
requests begin returning `403`. Restart the service after changing the value.
The Helm chart exposes the same setting as `s3AllowedBuckets`.

Entries are bucket names only (for example, `content-agent-intake`), not
`s3://` URIs, wildcards, or key prefixes. The application policy does not
restrict key prefixes, so use a dedicated intake bucket that contains no
unrelated or sensitive objects. Restrict the service IAM role to only the
required bucket and prefixes as defense in depth; IAM scoping does not replace
request authorization.

### VLM/LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `JA_VLM_BACKEND` | `nim` | VLM inference backend |
| `JA_VLM_MODEL` | `google/gemma-4-31b-it` | VLM model identifier |
| `JA_VLM_TEMPERATURE` | `1.0` | VLM sampling temperature |
| `JA_VLM_MAX_WORKERS` | `64` | Maximum concurrent prediction VLM requests per pipeline. NVCF staging deployments default to `4`. |
| `JA_RENDER_BACKEND` | `remote` | Rendering backend: `remote`, `warp`, `ovrtx`, or `mock`. Remote rendering resolves through `RENDER_ENDPOINT`; mock is test-only. |

### API Keys

| Variable | Fallback | Description |
|----------|----------|-------------|
| `JA_NVIDIA_API_KEY` | `NVIDIA_API_KEY` | NVIDIA inference API key |

### AWS (Optional)

| Variable | Description |
|----------|-------------|
| `AWS_CONFIG_FILE` | Path to `.env` file with AWS credentials |
| `AWS_ACCESS_KEY_ID` | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_DEFAULT_REGION` | AWS region |

---

## Typical Workflow

```
1. POST /pipeline/upload-usd        Upload USD file
       ↓
2. POST /pipeline                    Start pipeline (with session_id)
       ↓
3. GET /pipeline/{id}/events         Stream SSE for real-time progress
       ↓                             (or poll GET /pipeline/{id}/status)
4. GET /pipeline/{id}/results        Get final stats and download URLs
       ↓
5. GET /artifacts/{id}/report        View HTML report
   GET /artifacts/{id}/predictions   Download predictions JSONL
   GET /artifacts/{id}/articulation-report
                                      View articulation candidates HTML
   GET /artifacts/{id}/dataset       Download dataset JSONL
       ↓
6. POST /pipeline/{id}/regenerate    (Optional) Re-run steps with new prompt
       ↓
7. DELETE /sessions/{id}             Clean up when done
```

Or use the single-step shortcut:

```
1. POST /pipeline (with usd_file)   Upload + start in one call
       ↓
2. ... (same as steps 3-7 above)
```
