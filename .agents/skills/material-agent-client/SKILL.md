---
name: material-agent-client
description: Make requests to the Material Agent REST API service for VLM-based material assignment to 3D objects. Use when the user wants to use the Material Agent client, call the material agent service API, upload a USD file to the service, start or monitor a REST pipeline, generate reference images, download materialized results, or write a client script.
version: "0.1.0"
author: NVIDIA Content Agents
tags:
  - content-agents
  - material-agent
  - rest-api
  - client
  - usd
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Material Agent REST service, optional bearer-token authentication, curl or the bundled Python client, and provider credentials configured on the service for VLM, LLM, optional image generation, and optional embedding features.
---

# Material Agent Client

Use the Material Agent REST API to upload USD assets, run VLM material
assignment, monitor progress, and download materialized outputs.

## When to Use

- Use when the user asks for the Material Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a USD file, reuse an uploaded session,
  generate an AI reference image, submit custom materials, check pipeline
  status, stream events, or download results.
- Use the Docker deploy or quickstart skills first when no Material Agent
  service is running yet.
- Use `material-agent-cli` instead when the user wants to run the local CLI
  directly without the REST service.

## Limitations

- Keep credentials out of chat and commits. Use service-side environment
  variables or `MATERIAL_AGENT_TOKEN`; never ask the user to paste secrets.
- This skill calls an already running service. It does not build or start the
  service container.
- `render_num_workers=1` is recommended for one local OVRTX instance because
  it sets render worker and request concurrency to one. Increase it only when
  the render endpoint fronts multiple independent render service instances.
- Generated reference images depend on service-side image generation settings
  such as `MA_IMAGE_GEN_*` and the provider key for that backend.
- Prim clustering and embedding options depend on service-side embedding
  configuration.

## Prerequisites

- Material Agent service base URL, usually `http://localhost:8000`.
- `curl` and `jq` for shell examples, or Python plus the client in
  `references/client.py`.
- Optional bearer token supplied as `Authorization: Bearer <token>` or through
  `MATERIAL_AGENT_TOKEN`.
- A readable `.usd`, `.usda`, `.usdc`, or `.usdz` input file.
- Optional reference images, reference PDFs, or a custom materials ZIP that
  contains `materials.yaml` and its USD material library.

## Instructions

1. Confirm the service is reachable.
2. Choose one-step submission with `POST /pipeline`, or two-step submission
   with `POST /pipeline/upload-usd` followed by `POST /pipeline`.
3. Provide `user_email` only when the caller has one for tracking. When it is
   omitted or blank, the service uses `MA_DEFAULT_USER_EMAIL`, then
   `anonymous@nvidia.com` if that fallback is blank. Provide exactly one input
   source: `usd_file` or an existing `session_id`.
4. Pass `render_num_workers=1` for a single local OVRTX deployment.
5. Use `materials_zip` only when the user provides a custom material library.
6. Use generated reference images only after `GET /assets/{id}/input-render`
   is available.
7. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
8. Download outputs from the artifact endpoints after status is `completed`.

## Python Client

Copy or import the client from
`.agents/skills/material-agent-client/references/client.py`.

```python
from client import MaterialAgentClient

client = MaterialAgentClient(base_url="http://localhost:8000")

session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    reference_images=["reference.jpg"],
    render_num_workers=1,
    enable_prim_clustering=True,
    cluster_min_prims=25,
)

results = client.get_results(session_id)
print(results["download_urls"])
```

Re-run selected steps from a completed or failed session:

```python
client.regenerate(
    session_id,
    steps=["predict"],
    user_prompt="Prefer brushed aluminum for exposed frame components",
)

history = client.get_event_log(session_id)
print(history.get("total", len(history.get("events", []))))
```

Generate a reference image before the pipeline starts:

```python
session_id, status = client.run_and_monitor(
    usd_path="scene.usd",
    generated_reference_prompt="Satin red painted metal with black rubber tires",
    render_num_workers=1,
    user_email="user@example.com",
)
```

## curl Workflow

```bash
BASE_URL="http://localhost:8000"

curl -fsS "$BASE_URL/health" | jq .

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "render_num_workers=1" \
  -F "reference_images=@reference.jpg" | jq -r .session_id)

curl -fsS "$BASE_URL/pipeline/$SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$SESSION/events"

curl -fL -o output.usd "$BASE_URL/artifacts/$SESSION/output"
curl -fL -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"
curl -fL -o report.html "$BASE_URL/artifacts/$SESSION/report"
```

Replay stored progress events or re-run selected steps:

```bash
curl -fsS "$BASE_URL/pipeline/$SESSION/event-log" | jq .

curl -fsS -X POST "$BASE_URL/pipeline/$SESSION/regenerate" \
  -H "Content-Type: application/json" \
  -d '{"steps":["predict"],"user_prompt":"Prefer brushed aluminum"}' | jq .
```

Generated reference image flow:

```bash
SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline/upload-usd" \
  -F "usd_file=@scene.usd" | jq -r .session_id)

# Wait until the preview exists.
curl -fI "$BASE_URL/assets/$SESSION/input-render"

REF_ID=$(curl -fsS -X POST \
  "$BASE_URL/pipeline/$SESSION/generate-reference-image" \
  -F "prompt=Satin red painted metal with black rubber tires" | jq -r .reference_id)

curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "session_id=$SESSION" \
  -F "generated_reference_id=$REF_ID" \
  -F "render_num_workers=1" | jq .
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health/config | `GET /health`, `GET /config/vlm-models` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Generated refs | `POST /pipeline/{id}/generate-reference-image`, `DELETE /pipeline/{id}/generated-reference-image/{ref_id}`, `GET /assets/{id}/generated-ref`, `GET /assets/{id}/generated-ref/{ref_id}` |
| Artifacts | `GET /artifacts/{id}/output`, `/final-render`, `/predictions`, `/report`, `/optimization-report`, `/cluster-map`, `/cluster-report`, `/cluster-summary`, `/cluster-representatives` |
| Assets | `GET /assets/{id}/input-render`, `/previews`, `/preview/{name}`, `/references`, `/reference/{name}`, `/reference-pdfs`, `/reference-pdf/{name}` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`, `GET /sessions/usage`, `POST /sessions/admin/cleanup` |
| Materials | `GET /materials`, `GET /materials/icon/{name}`, `GET /materials/template` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data.

| Parameter | Required | Description |
|---|---|---|
| `usd_file` | Conditional | USD input file. Required unless `session_id` is provided. |
| `session_id` | Conditional | Existing session from `/pipeline/upload-usd`. |
| `user_email` | No | Optional tracking email. Omitted or blank uses `MA_DEFAULT_USER_EMAIL`, then `anonymous@nvidia.com`. |
| `reference_images` | No | One or more reference image files. |
| `reference_pdfs` | No | One or more reference PDF files. |
| `materials_zip` | No | ZIP containing `materials.yaml` and the USD material library. |
| `generated_reference_id` | No | Reference ID returned by the generated-reference flow. |
| `user_prompt` | No | Custom VLM prompt text. |
| `camera_views` | No | Camera directions such as `+x+y+z,-x-y-z`. |
| `vlm_model` | No | VLM model override. |
| `optimize_usd` | No | Enable USD optimization. |
| `vlm_max_workers` | No | Max parallel VLM workers. |
| `render_num_workers` | No | Render concurrency. Use `1` for one local OVRTX instance. |
| `enable_prim_clustering` | No | Enable image-based prim clustering before prediction. |
| `cluster_min_prims` | No | Minimum prim count before clustering runs. |
| `cluster_embedding_backend` | No | Embedding backend for clustering. |
| `cluster_embedding_model` | No | Embedding model for clustering. |
| `cluster_embedding_base_url` | No | Optional embedding endpoint base URL. Overrides are limited to hosted NVIDIA URLs or the deployment-configured `MA_CLUSTER_EMBEDDING_BASE_URL`. |
| `cluster_embedding_max_workers` | No | Maximum parallel embedding workers. |
| `cluster_embedding_batch_size` | No | Embedding batch size. |
| `cluster_max_size` | No | Maximum prims sharing one representative prediction. |
| `cluster_similarity_threshold_low` | No | Similarity threshold for low-complexity clusters. |
| `cluster_similarity_threshold_medium` | No | Similarity threshold for medium-complexity clusters. |
| `cluster_similarity_threshold_high` | No | Similarity threshold for high-complexity clusters. |
| `cluster_report` | No | Enable or disable the clustering HTML report. |
| `layer_only` | No | On initial submission, output only a material binding layer. |

`GET /pipeline/{id}/events` is the live SSE stream. Use
`GET /pipeline/{id}/event-log` to replay persisted progress events for a
completed, failed, or disconnected session.

`POST /pipeline/{id}/regenerate` accepts JSON.

| Parameter | Required | Description |
|---|---|---|
| `steps` | Yes | Pipeline steps to re-run from cache, such as `predict`, `apply`, or `render`. |
| `user_prompt` | No | Override the prompt for the regenerated run. |
| `layer_only` | No | When re-running `apply`, output only a material binding layer. |

`layer_only` is supported on both the initial multipart `POST /pipeline` and a
JSON regenerate/apply rerun. In the Python client, `enable_prim_clustering`,
`cluster_report`, and `layer_only` accept `True` or `False`; `None` omits the
corresponding field and leaves the service default unchanged.

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

## Output Format

Return a concise summary with:

- Service base URL and authentication mode, without printing tokens.
- Submitted endpoint and session ID.
- Input mode: direct file, uploaded session, generated reference, or custom
  materials.
- Current status and progress source: SSE or polling.
- Downloaded artifact paths or URLs for output USD, predictions, report, final
  render, optimization report, and cluster artifacts when available.
- Any service-side blocker such as missing credentials, upload size, renderer
  warm-up, or unavailable generated preview.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | The service is not running or the base URL is wrong. | Start the Material Agent service or correct `BASE_URL`. |
| Unauthorized | Bearer token is missing or wrong. | Set `MATERIAL_AGENT_TOKEN` locally or pass the correct header. |
| `413` upload response | Input exceeds the service upload limit. | Increase `MA_MAX_UPLOAD_SIZE_MB` on the service or use a smaller input. |
| `202` or missing results | Pipeline is still running. | Poll `/status` until `completed` or stream `/events`. |
| Generated reference fails | Input preview is not ready or image-gen backend credentials are missing. | Wait for `/assets/{id}/input-render` and check service-side `MA_IMAGE_GEN_*` settings. |
| Slow or failed renders | OVRTX is cold, overloaded, or over-concurrent. | Use `render_num_workers=1` for one local renderer and check the render endpoint health. |
| Custom materials rejected | ZIP is missing `materials.yaml` or bindings do not match the library. | Download `/materials/template` and preserve its structure. |
