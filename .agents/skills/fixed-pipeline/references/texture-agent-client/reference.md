---
name: texture-agent-client
description: "Fixed pipeline reference: Make requests to the Texture Agent REST API for texture generation. Use when the user explicitly requests the fixed pipeline service/client, uploads, REST monitoring, scoped prompts, artifact downloads, or a client script; use the agentic Content Workflow for unqualified texture tasks."
version: "0.1.5"
author: NVIDIA Content Agents
tags:
  - content-agents
  - texture-agent
  - rest-api
  - client
  - image-generation
tools:
  - Shell
  - Filesystem
  - Python
  - curl
  - jq
compatibility: Requires a running Texture Agent REST service, curl or the bundled Python client, a materialized USD input, and service-side image-generation and optional LLM credentials configured for the selected backend.
---

# Fixed Pipeline Reference: Texture Agent Client

This is the explicit fixed pipeline REST route, not the default agentic Content
Workflow.

Use the Texture Agent REST API to upload a materialized USD, discover materials,
generate image textures, apply them, and download the textured USDZ output.

## When to Use

- Use when the user asks for the Texture Agent service, API, REST workflow,
  Python client, or curl examples.
- Use when the user wants to upload a materialized USD, submit per-material
  prompts, keep texture generation scoped to specific materials, monitor
  pipeline status, or download generated textures and renders.
- Use `deploy-texture-agent-docker` first when no Texture Agent service is
  running yet.
- Use `texture-agent-cli` instead when the user wants the local CLI.

## Limitations

- Keep provider credentials out of chat and commits. They must be configured
  on the service; never ask the user to paste secrets.
- This skill calls an already running service. It does not build or start the
  service container.
- Input should already have material bindings, typically from Material Agent.
- The REST wire parameter is `material_textures_json`, a JSON-encoded string.
  The Python client exposes the same data as a `dict` named
  `material_textures`.
- Set `auto_prompt_enabled=false` when the user wants strict scope and only
  listed materials should be processed.
- Projection/reference-image texture editing uses `texture_backend=service`
  plus `texture_endpoint`; this is distinct from projection UV preparation,
  which still runs inside the pipeline before generation.
- A terminal service status and generated/accepted material counts establish
  artifact readiness, not visual acceptance or reference fidelity.

## Prerequisites

- Texture Agent service base URL, usually `http://localhost:8001`.
- `curl` and `jq` for shell examples, or Python plus
  `apps/texture_agent_service/client/client.py`.
- A materialized `.usd`, `.usda`, `.usdc`, or `.usdz` file.
- Service-side image-generation credentials such as `NVIDIA_API_KEY`,
  `GOOGLE_API_KEY`, or an endpoint-specific key.
- Optional LLM credentials when auto-prompt generation is enabled.

## Instructions

1. Confirm the service is reachable with `GET /health`.
2. Choose direct `POST /pipeline`, two-step upload, or S3 URI input.
3. Use exactly one input source: `usd_file`, `session_id`, or `s3_uri`.
4. Discover material names first when the user needs exact targeting. For a
   fresh USD with no completed session yet, run an auto-prompt discovery pass
   first, then resubmit the strict material map.
5. Submit `material_textures_json` as a JSON string for per-material prompts.
6. Use `auto_prompt_enabled=false` for strict listed-material scope.
7. For projection/reference-image backends, pass `texture_backend`,
   `texture_endpoint`, `backend_engine`, optional reference media, seed,
   strength, strict scope, and custom parameters.
8. Monitor with `GET /pipeline/{id}/events` for SSE or
   `GET /pipeline/{id}/status` for polling.
9. Download output USDZ, textures ZIP, manifest, materials JSON, and renders
   after status is `completed`.
10. Treat a later refinement as a candidate, not as an automatic replacement.
    Preserve the current accepted USDZ, manifest, textures, and preview under
    distinct paths with their SHA-256 digests before starting refinement.
11. Run the refinement candidate promotion gate below. Keep the prior accepted
    output when the comparison is blocked, incomplete, or worse.

## Refinement Candidate Promotion

Use this gate whenever a completed output is refined, regenerated, described
as more faithful to a reference, or altered by client/Codex post-processing.
Service completion alone must never promote the candidate.

1. Record the candidate origin as either the Texture Agent service session or
   client/Codex post-processing. Do not attribute post-processing to the
   service. Preserve the service session ID and all input/output digests.
2. Render the exact accepted baseline and candidate USD artifacts through the
   shared OVRTX USD render path with the same camera, framing, lighting,
   background, renderer identity, rendered-USD representation, and resolution.
   Preserve a digest-bound render-contract JSON file beside each image. Each
   contract records the exact source USD digest and retained OVRTX render
   metadata. If that matched production comparison cannot be produced, stop
   with a warning and keep the baseline; local, mock, generic remote, or preview
   renders cannot authorize promotion.
3. Compare both renders against the supplied reference for each requested
   material. Inspect base color, value, saturation, and material identity;
   valid files and map counts are not evidence that fidelity improved.
4. Inspect a base-color-only diagnostic and the full PBR result when normal,
   roughness, or ORM maps might change the appearance. This diagnostic may
   identify the responsible map, but silently dropping a requested map is not
   an acceptable fix.
5. Write a promotion receipt that names the baseline, candidate, reference,
   matching render evidence, per-material decision, map diagnostics, and the
   responsible stage. The decision must be one of `promote_candidate`,
   `reject_keep_baseline`, or `blocked_keep_baseline`.
6. Promote only when the evidence explicitly shows that every requested
   material improved or remained faithful without a new visual regression.
   Otherwise retain and return the prior accepted output, preserve the rejected
   candidate for diagnosis, and report the warning. Never overwrite the
   accepted files before this decision.
7. Validate the receipt before publishing or returning an accepted artifact:

   ```bash
   python .agents/skills/fixed-pipeline/references/texture-agent-client/scripts/validate_refinement_promotion.py \
     refinement_promotion.json
   ```

   A valid rejection or blocked comparison exits successfully only when its
   accepted-output digest matches the baseline. Candidate promotion exits
   successfully only when verified, matched OVRTX render contracts, map
   diagnostics, and all per-material findings prove `improved` or `preserved`
   fidelity. Exit code `1` means the receipt or evidence is invalid; exit code
   `2` means the receipt file could not be read.

### Promotion Receipt

The receipt uses schema `texture-agent-refinement-promotion.v1`. Artifact
objects contain `path` and `sha256`; paths must remain inside the receipt
directory. Absolute paths, parent traversal, and symlinks that resolve outside
that directory are rejected. The validator hashes every artifact rather than
trusting the asserted digest.

Required top-level fields are:

- `schema_version`, `decision`, `responsible_stage`, and `service_session_id`;
- `baseline`, `candidate`, and `accepted_output` artifact objects;
- `comparison_status`, either `complete` or `blocked`;
- for blocked comparisons, a non-empty `blockers` list;
- for complete comparisons, `comparison` with baseline/candidate/reference
  renders, digest-bound `baseline_render_contract` and
  `candidate_render_contract` JSON artifacts, unique `material_findings`, and
  base-color-only/full-PBR map diagnostics.

Each render-contract artifact uses schema
`texture-agent-ovrtx-render-contract.v1` and contains:

- `source_usd_sha256`, matching the baseline or candidate USD artifact;
- `render_path: "shared_ovrtx_usd"`;
- non-empty `camera`, `lighting`, and `background` objects;
- positive `resolution.width` and `resolution.height` values;
- `ovrtx_metadata` with an OVRTX-identifying `renderer`, lowercase SHA-256
  `rendered_usd_sha256` and `request_sha256` values, and a non-empty
  `rendered_usd_representation` identity.

The validator digest-verifies both contract files, verifies their source USD
bindings, and compares a canonical digest of their shared render conditions.
The source/rendered USD and request digests are request-specific and are not
part of that shared-condition comparison. Any OVRTX error, source mismatch, or
render-condition mismatch invalidates a complete comparison, so the workflow
must emit `blocked_keep_baseline` without a `comparison` object instead.

`responsible_stage` is one of `service`, `client_workflow`, or
`codex_postprocessing`. A complete `reject_keep_baseline` receipt must contain
at least one `regressed` material finding. A `blocked_keep_baseline` receipt
cannot contain a completed `comparison`; a complete receipt cannot contain
`blockers` or any blocked finding. Do not rename a candidate to the accepted
path to make a receipt pass; the file digests are authoritative. Schema changes
require shipping and invoking the validator version that names the new schema;
this validator intentionally fails closed on unknown versions.

## Python Client

```python
from apps.texture_agent_service.client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

session_id, status = client.run_and_monitor(
    usd_path="materialized_scene.usd",
    material_textures={
        "Steel_Carbon": {"prompt": "rusted steel", "opacity": 0.85},
        "Wood_Oak": {"prompt": "weathered oak planks", "opacity": 0.9},
    },
    auto_prompt_enabled=False,
)

client.download_output(session_id, "output.usdz")
client.download_textures(session_id, "./textures/")
```

Projection/reference-image backend run:

```python
from apps.texture_agent_service.client.client import TextureAgentClient

client = TextureAgentClient("http://localhost:8001")

session_id, status = client.run_and_monitor(
    usd_path="apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd",
    material_textures={
        "Aluminum_Matte": {
            "prompt": "matte aluminum with light scuffs",
            "opacity": 0.85,
        }
    },
    auto_prompt_enabled=False,
    texture_backend="service",
    texture_endpoint="http://REPLACE_WITH_TEXTURE_VARIATION_ENDPOINT",
    backend_engine="YOUR_ENGINE_OR_MODEL",
    backend_custom_parameters={"run_label": "manual-projection-run"},
    reference_image_uris=["file:///absolute/path/reference.png"],
    seed=11631,
    strength=0.85,
    strict_scope=True,
)
```

## curl Workflow

```bash
BASE_URL="http://localhost:8001"

curl -fsS "$BASE_URL/health" | jq .

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"rusted steel"}}' \
  | jq -r .session_id)

curl -fsS "$BASE_URL/pipeline/$SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$SESSION/events"

curl -fL -o output.usdz "$BASE_URL/artifacts/$SESSION/output"
curl -fL -o textures.zip "$BASE_URL/artifacts/$SESSION/textures"
curl -fL -o manifest.json "$BASE_URL/artifacts/$SESSION/manifest"
```

Fresh USD strict targeting:

```bash
# First pass: let the service discover material names and complete a session.
DISCOVERY_SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=true" \
  | jq -r .session_id)

# Wait for the discovery pass to reach completed status before fetching materials.
curl -fsS "$BASE_URL/pipeline/$DISCOVERY_SESSION/status" | jq .
curl -N "$BASE_URL/pipeline/$DISCOVERY_SESSION/events"
curl -fsS "$BASE_URL/artifacts/$DISCOVERY_SESSION/materials" | jq . > materials.json

# Second pass: use exact names from materials.json and strict scope.
STRICT_SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"rusted steel"}}' \
  | jq -r .session_id)
```

Projection/reference-image backend form fields:

```bash
BASE_URL="http://localhost:8001"
TEXTURE_VARIATION_ENDPOINT="http://REPLACE_WITH_TEXTURE_VARIATION_ENDPOINT"
BACKEND_ENGINE="YOUR_ENGINE_OR_MODEL"
MATERIAL_TEXTURES='{"Aluminum_Matte":{"prompt":"matte aluminum with light scuffs","opacity":0.85}}'

SESSION=$(curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@apps/texture_agent/data/examples/ladder/sources/usd/ladder.usd" \
  -F "auto_prompt_enabled=false" \
  -F "texture_backend=service" \
  -F "texture_endpoint=$TEXTURE_VARIATION_ENDPOINT" \
  -F "backend_engine=$BACKEND_ENGINE" \
  -F 'backend_custom_parameters_json={"run_label":"manual-projection-run"}' \
  -F 'reference_image_uris_json=["file:///absolute/path/reference.png"]' \
  -F "seed=11631" \
  -F "strength=0.85" \
  -F "strict_scope=true" \
  -F "material_textures_json=$MATERIAL_TEXTURES" \
  | jq -r .session_id)
```

Nested per-prim prompt map:

```bash
curl -fsS -X POST "$BASE_URL/pipeline" \
  -F "usd_file=@materialized_scene.usd" \
  -F "auto_prompt_enabled=false" \
  -F 'material_textures_json={"Steel_Carbon":{"prompt":"aged steel","per_prim":{"/World/Bolt":{"prompt":"scratched bolt head"}}}}' \
  | jq .
```

## Endpoint Reference

| Area | Endpoints |
|---|---|
| Health/API | `GET /health`, `GET /api`, `GET /` |
| Pipeline | `POST /pipeline/upload-usd`, `POST /pipeline`, `GET /pipeline/{id}/status`, `GET /pipeline/{id}/results`, `GET /pipeline/{id}/events`, `POST /pipeline/{id}/cancel`, `POST /pipeline/{id}/regenerate`, `GET /pipeline/{id}/event-log` |
| Artifacts | `GET /artifacts/{id}/materials`, `/manifest`, `/textures`, `/textures/{name}`, `/output`, `/renders`, `/renders/{name}`, `/preview/{name}` |
| Sessions | `GET /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}` |

## Key Pipeline Parameters

`POST /pipeline` accepts multipart form data. Exactly one of `usd_file`,
`session_id`, or `s3_uri` must be provided.

| Parameter | Required | Description |
|---|---|---|
| `usd_file` | Conditional | Materialized USD file. |
| `session_id` | Conditional | Existing uploaded session. |
| `s3_uri` | Conditional | Service-side S3 input. |
| `material_textures_json` | No | JSON-encoded string of per-material texture configuration. |
| `user_prompt` | No | Aesthetic direction for auto-prompt generation. |
| `auto_prompt_enabled` | No | Defaults to service behavior. Set `false` to process only listed materials. |
| `texture_backend` | No | Backend override. Use `service` for Texture Variation API projection backends. |
| `texture_endpoint` | Conditional | Texture Variation API endpoint, required when `texture_backend=service`. |
| `backend_engine` | No | Backend engine/model route hint. |
| `backend_custom_parameters_json` | No | JSON object of backend-specific parameters. |
| `reference_image_uris_json` | No | JSON list of global reference image URIs. |
| `reference_image_file` | No | Uploaded reference image file added to reference image conditioning. |
| `turntable_video_uri` | No | Global turntable video URI for backends that support it. |
| `multiview_image_uris_json` | No | JSON list of global multi-view image URIs. |
| `seed` | No | Texture backend seed override. |
| `strength` | No | Texture edit strength, 0.0 to 1.0. |
| `strict_scope` | No | Whether backend requests must preserve the selected target scope. |

Decoded `material_textures_json` is keyed by discovered material name. Each
value can include:

- `prompt`: text prompt describing the desired texture.
- `opacity`: optional blend opacity.
- `per_prim`: optional nested map keyed by prim path for prim-specific
  overrides.

Use `GET /artifacts/{id}/materials` to discover exact material names from a
completed session before submitting a strict map. `GET /pipeline/{id}/results`
is also completed-only and can confirm artifact URLs, but it is not available
before the first run. For a fresh USD, first submit a discovery pass with
`auto_prompt_enabled=true`, wait for it to complete, fetch
`GET /artifacts/{id}/materials`, then submit the exact
`material_textures_json` keys in a second run with `auto_prompt_enabled=false`.

Status values are `pending`, `running`, `completed`, `failed`, `cancelled`,
and `cancelling`.

## Output Format

Return a concise summary with:

- Service base URL and input mode.
- Session ID, status, and progress source.
- Whether auto-prompting was enabled or strict listed-material scope was used.
- Submitted material names and any `per_prim` overrides.
- Projection backend endpoint/engine, conditioning fields, seed/strength, and
  strict scope when used.
- Downloaded artifact paths or URLs for output USDZ, textures ZIP, manifest,
  materials JSON, event log, and renders.
- For refinement, baseline/candidate/reference paths and digests, matched-render
  settings, per-material comparison, map diagnostics, promotion decision, and
  whether the responsible stage was the service or client/Codex processing.
- Any blocker such as missing image-generation credentials, unmatched material
  names, backend capability mismatch, degraded maps, low coverage, portability
  diagnostics, unavailable matched renders, a rejected visual candidate, upload
  size, or non-terminal status.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Connection refused | Service is not running or base URL is wrong. | Start the Texture Agent service or correct `BASE_URL`. |
| `413` upload response | Input exceeds service upload limit. | Increase `TA_MAX_UPLOAD_SIZE_MB`, use a smaller file, or submit an S3 URI. |
| Results return `202` | Pipeline is still running. | Poll `/status` until `completed` or stream `/events`. |
| No textures generated for a material | The material name did not match discovered names, or strict mode skipped it. | Fetch `/artifacts/{id}/materials`, then resubmit exact keys. |
| Unexpected extra textures | Auto-prompting processed unlisted materials. | Set `auto_prompt_enabled=false`. |
| `material_textures_json` parse error | The REST form value was not valid JSON text. | Quote it as one JSON string, or use the Python client dict abstraction. |
| Image generation fails | Service-side backend credentials or endpoint are missing. | Check service logs and the configured image-generation backend. |
| Projection backend endpoint is rejected | `texture_endpoint` is missing while `texture_backend=service`. | Provide a reachable Texture Variation API endpoint. |
| Backend reports unsupported conditioning | Reference image, turntable, or multi-view fields exceed backend capabilities. | Retry with supported conditioning or use a backend that supports the requested media. |
| Projection backend reports missing albedo | The required base-color map was not returned. | Treat the run as failed and inspect `/artifacts/{id}/manifest`. |
| Optional maps are degraded | Backend omitted normal or ORM channels. | Texture Agent records degraded channels and synthesizes or packs maps when possible. |
| Low coverage is reported | Backend coverage metadata is below threshold for the selected target. | Inspect coverage/mask/debug artifacts and retry with clearer scope or reference images. |
| Output portability fails | Generated USD texture references are not package-local. | Inspect manifest portability diagnostics before sharing the output package. |
| A refinement is paler, brighter, or less reference-faithful | The candidate passed artifact validation but regressed visually, or client-side post-processing changed the service result. | Reject the candidate, keep the baseline, compare same-camera renders and base-color-only versus full-PBR diagnostics, and report the responsible stage. |
