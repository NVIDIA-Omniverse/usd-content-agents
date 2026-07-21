---
name: physics-agent-client
description: Make requests to the Physics Agent REST API service for VLM-based physics and component classification of 3D USD assets. Use when the user wants to call the Physics Agent service API, upload or reference a USD file, run the REST classification pipeline, use prediction-only, tuning, or refine-loop routes, monitor status, download predictions, simulation-ready USD, or refinement artifacts, or write a client script.
version: "0.1.1"
author: NVIDIA Content Agents
tags:
  - content-agents
  - physics-agent
  - rest-api
  - client
  - usd
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Physics Agent REST service, optional bearer-token authentication, curl or the bundled Python client, and service-side VLM, rendering, optimizer, S3, and optional tuning/refine dependencies configured for the requested route.
---

# Physics Agent Client

Use the Physics Agent REST API to classify USD components, generate
`predictions.jsonl`, download a simulation-ready USD with physics schemas
applied, tune authored physics parameters, or run the iterative refine loop.

## When to Use

- Use when the user asks for the Physics Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a USD file, submit an S3 URI, run the full
  classification pipeline, check status, stream events, or download artifacts.
- Use `/predict` when the user explicitly wants prediction-only behavior.
- Use `/tune` only when the user has a physics-authored USD and wants a
  service-side tuning run.
- Use `/refine` only when the user has a physics-authored USD or completed
  `/pipeline` session and wants the service to run the iterative
  tune-judge-scenario-refine loop.
- Use `physics-agent-cli` instead when the user wants to run the local CLI.

## Limitations

- Keep credentials out of chat and commits. Use service-side configuration or
  `PHYSICS_AGENT_TOKEN`; never ask the user to paste secrets.
- This is a client-only workflow for an already running service. If
  `GET /health` is unreachable, stop client execution. Do not build, start,
  restart, or reconfigure the service, and do not mutate Docker, Compose, Brev,
  or other deployment infrastructure from this skill. Route local deployment
  work to `deploy-physics-agent-docker` and Brev deployment work to
  `deploy-physics-agent-brev`.
- The main guidance is for `/pipeline`; `/predict`, `/tune`, and `/refine` are
  related API families with separate session kinds and cancel endpoints.
- The service `/refine` route builds judge/refiner models from server-side
  `PA_REFINE_BACKEND` / `PA_REFINE_MODEL` config. Callers select no backend and
  submit no provider credentials through the request.
- Use deinstance only when an error literally identifies an instance proxy or
  USD inspection confirms the target is an instance proxy. Use split only when
  one mesh is confirmed to contain multiple disjoint components. Do not apply
  either flag to generic VLM-output, prediction-schema, or apply-physics
  failures.
- A synthetic `fake` engine run is test infrastructure only. It is never
  production, release, simulator, tuning-quality, or acceptance evidence.
- OVRTX can return `gpu_initialized=false` while warming up even when the HTTP
  service is reachable.

## Prerequisites

- Physics Agent service base URL, usually `http://localhost:8000`.
- `curl` and `jq` for shell examples, or Python plus
  `apps/physics_agent_service/client/client.py`.
- Optional bearer token supplied as `Authorization: Bearer <token>` or through
  `PHYSICS_AGENT_TOKEN`.
- A local USD file, an uploaded session, or an S3 URI readable by the service.
- Service-side provider credentials and render/optimizer endpoints configured
  for the requested run.
- For `/refine`, a scenario YAML body, a natural-language `user_prompt`, and a
  physics-authored USD source: `physics_usd`, `s3_uri`, or `source_session_id`
  from a completed `/pipeline` run.

## Instructions

1. Run the curl workflow's `physics_client_health_preflight` before any
   submission. Treat its nonzero result as a hard stop: do not provision or
   change infrastructure, and report the appropriate named deployment-skill
   handoff.
2. Choose `/pipeline`, `/predict`, `/tune`, or `/refine` and follow that route's
   input-source contract below. Form names are route-specific.
3. Before the first `POST`, start a request ledger. Route every submission
   through a shared wrapper that appends the attempt's ordinal, endpoint, input
   mode, HTTP status or transport error, response body, and session ID when
   available before returning or re-raising an error.
4. For `/pipeline`, choose direct upload, two-step upload, or S3 input. Use one
   of `usd_file`, `session_id`, or `s3_uri`.
5. Add optimizer flags only for confirmed instancing, instance-proxy authoring,
   disjoint mesh splitting, or deduplication needs.
6. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
7. Download predictions, report, dataset, and output USD after status is
   `completed`.
8. For prediction-only, tuning, or refine flows, use the matching `/predict`,
   `/tune`, or `/refine` status, events, results, and cancel endpoints.
9. For `/refine` from an apply-physics output, first wait for `/pipeline` to
   complete, then pass that pipeline ID as `source_session_id`.
10. Reconcile every accepted execution-start `POST` with terminal service
    status before reporting counts. A retry or regenerate is a new execution
    attempt even when it reuses a session ID; capture terminal evidence before
    that ID starts another generation. Exclude accepted upload-only requests.

### Submission Accounting

Record an attempt before errors escape the submission call. The curl wrapper
captures the response and status before returning nonzero and rejects curl
`--fail`/`-f`, which can suppress the response body. Python callers must wrap
each low-level `POST` helper separately, capture `HTTPError.response.status_code`
and `HTTPError.response.text` (or a transport exception), append the entry, then
re-raise. Do not count `run_and_monitor` as one call: S3 or `upload_first` mode
can issue both `upload_usd` and `start_pipeline`.

## Python Client

```python
import time

from apps.physics_agent_service.client.client import PhysicsAgentClient

client = PhysicsAgentClient(base_url="http://localhost:8000")


def wait_for_terminal(get_status, session_id, timeout_s=1800):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = get_status(session_id)
        if status["status"] in {"completed", "failed", "cancelled"}:
            return status
        time.sleep(2)
    raise TimeoutError(f"session {session_id} did not reach a terminal status")

session_id = client.start_pipeline(
    usd_path="/path/to/scene.usdz",
    user_prompt="Focus on identifying furniture parts",
    render_backend="remote",
)
status = wait_for_terminal(client.get_status, session_id)

print(session_id, status)
```

Use S3 input mode for large files when the service can download the asset:

```python
upload_session = client.upload_usd(s3_uri="s3://your-bucket/path/to/scene.usdz")
session_id = client.start_pipeline(session_id=upload_session)
status = wait_for_terminal(client.get_status, session_id)
```

Use the dedicated helpers for non-pipeline route families; do not send
tune/refine work through `/pipeline`:

```python
predict_session = client.start_predict(session_id=session_id)
predict_status = wait_for_terminal(client.get_predict_status, predict_session)
predict_results = (
    client.get_predict_results(predict_session)
    if predict_status["status"] == "completed"
    else None
)

tune_session = client.start_tune(
    source_session_id=session_id,
    scenario_yaml_path="apps/physics_agent/configs/tuning/drop_settle.yaml",
    optimizer="botorch",
    seed=42,
)
tune_status = wait_for_terminal(client.get_tune_status, tune_session)
tune_results = (
    client.get_tune_results(tune_session)
    if tune_status["status"] == "completed"
    else None
)

refine_session = client.start_refine(
    source_session_id=session_id,
    scenario_yaml_path="apps/physics_agent/configs/tuning/drop_settle.yaml",
    user_prompt="make the object settle on the target surface",
    optimizer="botorch",
    score_threshold=0.9,
    seed=42,
)
refine_status = wait_for_terminal(client.get_refine_status, refine_session)
refine_results = (
    client.get_refine_results(refine_session)
    if refine_status["status"] == "completed"
    else None
)
tuned_usd = (
    client.download_refine_artifact(
        refine_session,
        "final/tuned_physics.usda",
    )
    if refine_status["status"] == "completed"
    else None
)
```

## curl Workflow

Run this block from the repo root as one shell script. Its health failure exits
before `physics_client_submit` can issue the first `POST`. The remaining curl
examples reuse the sourced helpers.

```bash
set -euo pipefail

BASE_URL="http://localhost:8000"
REQUEST_LEDGER="${REQUEST_LEDGER:-physics-agent-request-ledger.jsonl}"
PHYSICS_CLIENT_CONNECT_TIMEOUT_S=10
PHYSICS_CLIENT_REQUEST_TIMEOUT_S=600
PHYSICS_CLIENT_POLL_TIMEOUT_S=1800
CURL_TIMEOUTS=(--connect-timeout "$PHYSICS_CLIENT_CONNECT_TIMEOUT_S" --max-time "$PHYSICS_CLIENT_REQUEST_TIMEOUT_S")
source .agents/skills/physics-agent-client/scripts/request_helpers.sh

if ! physics_client_health_preflight; then
  exit 1
fi

PIPELINE_RESPONSE=$(physics_client_submit "/pipeline" "usd_file" \
  -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd")
SESSION=$(jq -er '.session_id' <<<"$PIPELINE_RESPONSE")
PIPELINE_STATUS=$(physics_client_poll_until_terminal pipeline "$SESSION")
if [[ "$PIPELINE_STATUS" != "completed" ]]; then
  printf 'Pipeline ended with status: %s\n' "$PIPELINE_STATUS" >&2
  exit 1
fi

curl "${CURL_TIMEOUTS[@]}" -fL -o predictions.jsonl "$BASE_URL/artifacts/$SESSION/predictions"
curl "${CURL_TIMEOUTS[@]}" -fL -o report.html "$BASE_URL/artifacts/$SESSION/report"
curl "${CURL_TIMEOUTS[@]}" -fLOJ "$BASE_URL/artifacts/$SESSION/output-usd"
```

Optimizer examples:

```bash
# Deinstance to fix instance-proxy apply_physics failures.
OPTIMIZED_RESPONSE=$(physics_client_submit "/pipeline" "usd_file+deinstance" \
  -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true")
OPTIMIZED_SESSION=$(jq -er '.session_id' <<<"$OPTIMIZED_RESPONSE")
OPTIMIZED_STATUS=$(physics_client_poll_until_terminal pipeline "$OPTIMIZED_SESSION")
[[ "$OPTIMIZED_STATUS" == "completed" ]] || exit 1
jq . <<<"$OPTIMIZED_RESPONSE"

# Deinstance and split when one mesh must become separate components.
SPLIT_RESPONSE=$(physics_client_submit "/pipeline" "usd_file+deinstance+split" \
  -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@scene.usd" \
  -F "optimize_usd=true" \
  -F "enable_deinstance=true" \
  -F "enable_split=true")
SPLIT_SESSION=$(jq -er '.session_id' <<<"$SPLIT_RESPONSE")
SPLIT_STATUS=$(physics_client_poll_until_terminal pipeline "$SPLIT_SESSION")
[[ "$SPLIT_STATUS" == "completed" ]] || exit 1
jq . <<<"$SPLIT_RESPONSE"
```

### Predict Workflow

For a prediction-only upload, `/predict` uses the multipart file field
`usd_file`:

```bash
PREDICT_RESPONSE=$(physics_client_submit "/predict" "usd_file" \
  -X POST "$BASE_URL/predict" \
  -F "usd_file=@scene.usd" \
  -F "user_prompt=classify the physical properties")
PREDICT_SESSION=$(jq -er '.session_id' <<<"$PREDICT_RESPONSE")

PREDICT_STATUS=$(physics_client_poll_until_terminal predict "$PREDICT_SESSION")
if [[ "$PREDICT_STATUS" != "completed" ]]; then
  printf 'Predict ended with status: %s\n' "$PREDICT_STATUS" >&2
  exit 1
fi
curl "${CURL_TIMEOUTS[@]}" -fsS "$BASE_URL/predict/$PREDICT_SESSION/results" | jq .
```

### Tune Workflow

For a direct physics-authored USD upload, `/tune` uses `physics_usd`, not
`usd_file`. Supply at least one of `scenario_yaml` or `user_prompt`:

```bash
TUNE_RESPONSE=$(physics_client_submit "/tune" "physics_usd" \
  -X POST "$BASE_URL/tune" \
  -F "physics_usd=@physics.usd" \
  -F "scenario_yaml=<apps/physics_agent/configs/tuning/drop_settle.yaml" \
  -F "optimizer=botorch" \
  -F "engine=ovphysx" \
  -F "max_trials=30" \
  -F "seed=42")
TUNE_SESSION=$(jq -er '.session_id' <<<"$TUNE_RESPONSE")

TUNE_STATUS=$(physics_client_poll_until_terminal tune "$TUNE_SESSION")
if [[ "$TUNE_STATUS" != "completed" ]]; then
  printf 'Tune ended with status: %s\n' "$TUNE_STATUS" >&2
  exit 1
fi
curl "${CURL_TIMEOUTS[@]}" -fsS "$BASE_URL/tune/$TUNE_SESSION/results" | jq .
```

### Refine Workflow

Use `/refine` after `/pipeline` has produced an apply-physics `output_usd`, or
when the caller already has a physics-authored USD. The route accepts exactly
one source: `source_session_id`, `physics_usd`, or `s3_uri`. A direct upload
uses the `physics_usd` form name, never `usd_file`; this example reuses a
completed pipeline session instead.

```bash
# Reuse a completed pipeline session so refine consumes its output_usd.
REFINE_RESPONSE=$(physics_client_submit "/refine" "source_session_id" \
  -X POST "$BASE_URL/refine" \
  -F "source_session_id=$SESSION" \
  -F "scenario_yaml=<apps/physics_agent/configs/tuning/drop_settle.yaml" \
  -F "user_prompt=make the object settle on the target surface" \
  -F "optimizer=botorch" \
  -F "score_threshold=0.9" \
  -F "seed=42")
REFINE_SESSION=$(jq -er '.session_id' <<<"$REFINE_RESPONSE")

REFINE_STATUS=$(physics_client_poll_until_terminal refine "$REFINE_SESSION")
if [[ "$REFINE_STATUS" != "completed" ]]; then
  printf 'Refine ended with status: %s\n' "$REFINE_STATUS" >&2
  exit 1
fi
curl "${CURL_TIMEOUTS[@]}" -fsS "$BASE_URL/refine/$REFINE_SESSION/results" | jq .
curl "${CURL_TIMEOUTS[@]}" -fL -o refine_summary.json \
  "$BASE_URL/refine/$REFINE_SESSION/artifacts/refine_summary.json"
curl "${CURL_TIMEOUTS[@]}" -fL -o tuned_physics.usda \
  "$BASE_URL/refine/$REFINE_SESSION/artifacts/final/tuned_physics.usda"
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health | `GET /health` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Predict-only | `POST /predict`, `GET /predict/{id}/status`, `GET /predict/{id}/results`, `GET /predict/{id}/events`, `POST /predict/{id}/cancel` |
| Artifacts | `GET /artifacts/{id}/predictions`, `/report`, `/dataset`, `/output-usd` |
| Tune | `POST /tune`, `GET /tune/{id}/status`, `GET /tune/{id}/results`, `GET /tune/{id}/events`, `POST /tune/{id}/cancel`, `GET /tune/{id}/artifacts/{name}` |
| Refine | `POST /refine`, `GET /refine/{id}/status`, `GET /refine/{id}/results`, `GET /refine/{id}/events`, `POST /refine/{id}/cancel`, `GET /refine/{id}/artifacts/{name}` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `usd_file` | Conditional |  | USD input file. Required unless `session_id` or `s3_uri` is provided. |
| `session_id` | Conditional |  | Existing session from `/pipeline/upload-usd`. |
| `s3_uri` | Conditional |  | Service-side S3 input. |
| `user_prompt` | No |  | Custom classification guidance. |
| `render_backend` | No | `remote` | Backend name forwarded unchanged; validated by the server against the current canonical registry. |
| `optimize_usd` | No | `false` | Run Scene Optimizer before rendering and prediction. |
| `enable_deinstance` | No | `true` when optimizing | Deinstance optimized USDs; required for instance-proxy authoring failures. |
| `enable_split` | No | `false` | Split disjoint pieces in one mesh into separate components. |
| `enable_deduplicate` | No | `false` | Collapse repeated identical geometry and restore by correspondence. |

At least one of `usd_file`, `session_id`, or `s3_uri` must be provided. If a
request supplies multiple sources, the service selects `session_id`, then
`s3_uri`, then `usd_file`; client workflows should still send one source so the
chosen input is explicit and auditable. When `optimize_usd=true`, enable at
least one optimizer operation.

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

## Key Predict Parameters

`POST /predict` accepts multipart form data and auto-selects prediction-only
Mode A or USD-preparation Mode B.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `usd_file` | Conditional |  | Mode B USD upload. The multipart file field is named `usd_file`. |
| `session_id` | Conditional |  | Existing service session. May be paired with `dataset_path` as an override. |
| `s3_uri` | Conditional |  | Mode B service-side S3 USD input. |
| `dataset_path` | Conditional |  | Mode A server-local prepared `dataset.jsonl`; allowed alone or with `session_id`. |
| `user_prompt` | No |  | Custom prediction guidance used in Mode B. |
| `optimize_usd` | No | `false` | Run Scene Optimizer before Mode B dataset preparation. |
| `enable_deinstance` | No | `true` | Deinstance when optimizing Mode B input. |
| `enable_split` | No | `false` | Split disjoint meshes when optimizing Mode B input. |
| `enable_deduplicate` | No | `false` | Deduplicate repeated geometry when optimizing Mode B input. |

At least one input source is required. Provide at most one primary source from
`usd_file`, `session_id`, and `s3_uri`. `dataset_path` may be used alone or with
`session_id`; it is incompatible with `usd_file` and `s3_uri`. Mode B uses the
same server-validated `render_backend` documented in Key Pipeline Parameters
and the optimizer controls above; Mode A ignores those preparation controls.

## Key Tune Parameters

`POST /tune` accepts multipart form data for one tuning pass.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `physics_usd` | Conditional |  | Physics-authored USD upload. The multipart file field is named `physics_usd`, not `usd_file`. |
| `source_session_id` | Conditional |  | Completed `/pipeline` session whose apply-physics `output_usd` should be tuned. |
| `s3_uri` | Conditional |  | Service-side S3 input pointing to a physics-authored USD. |
| `scenario_yaml` | Conditional |  | Scenario YAML text. Either this field or `user_prompt` is required. |
| `user_prompt` | Conditional |  | Natural-language scenario request. Either this field or `scenario_yaml` is required. |
| `reference_images` | No |  | Repeatable image-upload field for judge evidence. |
| `reference_videos` | No |  | Repeatable video-upload field for judge evidence. |
| `optimizer` | No | `auto` | `auto`, `botorch`, `random`, or `cma-es`. |
| `engine` | No | `ovphysx` | Simulation engine selected by the service. |
| `max_trials` | No | `30` | Optimizer trial budget. |
| `seed` | No | `42` | Seed for optimizer and backend. |
| `enable_judge` | No | `true` | Run the final VLM-as-judge pass. |

Exactly one of `physics_usd`, `source_session_id`, or `s3_uri` must be
provided. When both `scenario_yaml` and `user_prompt` are supplied, explicit
YAML fields override interpreted prompt fields.

## Key Refine Parameters

`POST /refine` accepts multipart form data.

| Parameter | Required | Default | Description |
|---|---|---|---|
| `physics_usd` | Conditional |  | Physics-authored USD upload. Required unless `source_session_id` or `s3_uri` is provided. |
| `source_session_id` | Conditional |  | Completed `/pipeline` session whose apply-physics `output_usd` should be refined. |
| `s3_uri` | Conditional |  | Service-side S3 input pointing to a physics-authored USD. |
| `scenario_yaml` | Yes |  | Tuning scenario YAML text, commonly submitted with `-F scenario_yaml=<file.yaml`. |
| `user_prompt` | Yes |  | Natural-language target behavior for judge/refine calls. |
| `optimizer` | No | `botorch` | `botorch`, `auto`, `random`, or `cma-es`. |
| `score_threshold` | No | `0.9` | Judge approval threshold in `[0, 1]`. |
| `max_trials` | No | `30` | Tuning trials per refine iteration. |
| `max_iterations` | No | `5` | Hard cap on tune/judge/refine iterations. |
| `seed` | No | `42` | Seed for optimizer and backend. |

Exactly one of `physics_usd`, `source_session_id`, or `s3_uri` must be
provided. Prefer `source_session_id` when `/pipeline` just produced the
apply-physics output.

## Output Format

Return a concise summary with:

- Service base URL and authentication mode, without printing tokens.
- Endpoint family used: `/pipeline`, `/predict`, `/tune`, or `/refine`.
- A request ledger with one entry per `POST`: attempt number, endpoint, input
  mode, observed HTTP status or transport error, response body, and returned
  session ID when the service accepted it. Record the entry before propagating
  any curl or Python exception.
- Exact totals that classify every `POST` attempt once by transport outcome:
  accepted responses (`2xx`), rejected requests (`4xx`, including rejected
  pre-session requests), server failures (`5xx`), unexpected HTTP responses
  (`1xx` or `3xx`), or transport failures (no HTTP status). Record session-ID
  presence separately and validate it against the route contract; unexpected
  presence or absence is a contract anomaly, not a different bucket. An attempt
  without terminal status evidence is not a failed session. Never infer status.
- Exact terminal counts from service evidence for execution attempts keyed by
  `(route family, session ID, ledger attempt or generation)`: successful
  (`completed`), `failed`, and `cancelled`. Retries and `/regenerate` calls count
  separately even when they reuse an ID; capture each terminal status before it
  is overwritten. `/pipeline/upload-usd` is an accepted upload-only request,
  while each later execution-start `POST` contributes its own outcome.
- Session ID, execution attempt/generation, input mode, terminal status, and
  progress source for each execution attempt.
- Optimizer flags used and why they were needed.
- Downloaded artifact paths or URLs for predictions, report, dataset, output
  USD, tune artifacts, and refine artifacts when applicable.
- Any blocker such as missing service credentials, upload size, S3 access,
  renderer warm-up, server-side refine provider config, or non-terminal status.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused, timeout, or unreachable `/health` | No usable already-running service is available at `BASE_URL`. | Stop client execution and verify `BASE_URL`. Do not provision or mutate infrastructure; hand local deployment to `deploy-physics-agent-docker` or Brev deployment to `deploy-physics-agent-brev`. |
| Unauthorized | Bearer token is missing or wrong. | Set `PHYSICS_AGENT_TOKEN` locally or pass the correct header. |
| `413` upload response | Input exceeds the service upload limit. | Use a smaller file or submit an S3 URI. If the limit must change, hand the request to `deploy-physics-agent-docker`, `deploy-physics-agent-brev`, or the service owner; do not reconfigure it from this client skill. |
| Results return `202` | The predict, pipeline, tune, or refine session is still running. | Poll the matching `/status` route to `completed`, `failed`, or `cancelled` (or stream `/events`) before requesting results. |
| `/tune` rejects an upload with "exactly one" source required | The request used pipeline's `usd_file` field, so `/tune` received no physics source. | Record the rejected attempt and resubmit with `physics_usd`; do not rename fields across route families. |
| Missing `physical_properties`, invalid VLM output/schema, or no schemas applied | Prediction content, `output_key`, or prediction prim paths do not satisfy the apply-physics contract. | Inspect `predictions.jsonl`, raw VLM output, `classification.physical_properties`, `output_key`, and target prim paths. Do not enable optimizer flags unless separate geometry evidence requires them. |
| Error literally identifies an instance proxy, or USD inspection confirms one | Physics schemas cannot be authored on that instance proxy. | Re-run `/pipeline` with `optimize_usd=true` and `enable_deinstance=true`. |
| One confirmed mesh contains multiple disjoint components | Deinstance can make a prim writable but does not separate disconnected geometry. | Add `enable_split=true` only when separate component entries are required. |
| `/refine` rejects `source_session_id` | The referenced pipeline did not complete or has no `output_usd`. | Wait for `/pipeline/{id}/status` to be `completed`, then retry. |
| `/refine` fails before first iteration | Tuning/OvPhysX dependencies, `PA_REFINE_BACKEND`, `PA_REFINE_MODEL`, or provider credentials are missing. | Use a tuning-enabled image and select a provider registered for both chat and VLM. |
| OVRTX health shows `gpu_initialized=false` | Renderer is still warming. | Wait and re-check the render sidecar health before treating it as failed. |
