# Geometry Benchmark Bundles

The Content Agents Dashboard is an artifact consumer. It does not run Geometry,
render USD, or calculate benchmark verdicts. A Geometry publisher supplies one
complete run directory whose `score/suite_result.json` is authoritative.

## Layout

```text
<artifact-root>/<workflow>/<run-id>/
|-- benchmark_run.json
|-- bundle/
|   |-- run.json
|   `-- assets/...
|-- score/
|   `-- suite_result.json
`-- evaluation/
    `-- material_evaluation.json
```

The score and evaluation documents are optional. When present, they are part of
the consumer-derived, digest-bound artifact set.

The adapter accepts `cad_to_simready`, `cad-to-simready`, and
`geometry-cad-to-simready` as exact published workflow IDs and groups them under
the CAD-to-SimReady dashboard tab. It accepts `geometry-public-cad` as the
digest-bound public prompt-to-CAD comparison workflow and groups it under the
CAD Agent Benchmark tab. The selected ID must agree in the index, run manifest,
bundle, and every scored case. These IDs do not define contracts for other
Geometry benchmark families.

## Run Manifest

`benchmark_run.json` uses `content-agent-benchmark-run.v1`. In addition to the
standard identity, status, timestamp, and case count, its `metadata` should
carry:

- `git`: target branch, commit, and dirty state.
- `provenance`: exact target, harness, and dataset fingerprints.
- `execution.runner_hardware`: OS, CPU, memory, and GPU records.
- `execution.renderers`: workflow and benchmark-evidence deployments. OVRTX
  settings belong under `benchmark_evidence.settings`.
- `execution.usage`: input, cached-input, output, and total tokens plus the
  versioned cost estimate and its explicit estimate status.
- `vlm`: backend and model when a model participated in the workflow.

The dashboard displays these records as provenance, environment, runtime,
token, and estimated-cost data. Cost estimates are never represented as actual
billing unless their own record says so.

## Run Bundle

Geometry `bundle/run.json` uses `content-agent-benchmark-bundle.v1`. It retains
the generic bundle fields consumed by the benchmark collector and adds one
namespaced extension per collected asset:

```json
{
  "schema_version": "content-agent-benchmark-bundle.v1",
  "run_id": "geometry-cad-to-simready-20260804-001",
  "workflow": "geometry-cad-to-simready",
  "label": "CAD-to-SimReady canary",
  "created_at": "2026-08-04T12:00:00Z",
  "source": "external-evidence-publisher",
  "git": { "branch": "main", "commit": "01234567", "dirty": false },
  "assets": [
    {
      "asset_id": "fixture_valve",
      "name": "Fixture valve",
      "status": "conditional",
      "tags": ["public_canary", "rigid_body"],
      "source_usd": "assets/fixture_valve/source.usdc",
      "output_usd": "assets/fixture_valve/output.usdc",
      "metrics": {
        "runtime_seconds": 42.5,
        "input_tokens": 1200,
        "cached_input_tokens": 300,
        "output_tokens": 240,
        "total_tokens": 1440
      },
      "renders": {
        "front": "assets/fixture_valve/renders/front.png"
      },
      "geometry": {
        "schema_version": "content-agent-benchmark-geometry.v1",
        "benchmark_family": "cad_to_simready",
        "outcome": "conditional",
        "handoff_ready": "conditional",
        "artifacts": [
          {
            "kind": "source_usd",
            "label": "Source USD",
            "path": "assets/fixture_valve/source.usdc",
            "sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "media_type": "application/octet-stream"
          },
          {
            "kind": "output_usd",
            "label": "Output USD",
            "path": "assets/fixture_valve/output.usdc",
            "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "media_type": "application/octet-stream"
          },
          {
            "kind": "content_agents_manifest",
            "label": "Content Agents manifest",
            "path": "assets/fixture_valve/content_agents_manifest.json",
            "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "media_type": "application/json"
          },
          {
            "kind": "geometry_validation_evidence",
            "label": "Geometry validation evidence",
            "path": "assets/fixture_valve/geometry_validation_evidence.json",
            "sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "media_type": "application/json"
          },
          {
            "kind": "ovrtx_render_manifest",
            "label": "OVRTX render manifest",
            "path": "assets/fixture_valve/renders/render_manifest.json",
            "sha256": "9999999999999999999999999999999999999999999999999999999999999999",
            "media_type": "application/json"
          }
        ],
        "render_evidence": [
          {
            "view": "front",
            "image": "assets/fixture_valve/renders/front.png",
            "image_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "camera": "assets/fixture_valve/renders/front_camera.json",
            "camera_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "renderer": "ovrtx",
            "render_quality": "final",
            "ovrtx_render_mode": "pt",
            "ovrtx_num_sensor_updates": 500,
            "active_aov": "LdrColor",
            "width": 640,
            "height": 640,
            "fallback": false,
            "elapsed_seconds": 12.5
          }
        ],
        "findings": [
          {
            "severity": "warning",
            "title": "Downstream validation pending",
            "detail": "Runtime cooking remains owned by the SimReady stage."
          }
        ]
      }
    }
  ]
}
```

Public CAD comparison assets may additionally carry a non-empty `prompt`. The
publisher places the native same-model OVRTX view in `references`, the CAD
Agent view in `geometry.render_evidence`, and lane metrics and outcomes in the
asset and scored-case metrics. The authoritative case status represents CAD
Agent hard-gate plus visual-review quality; pairwise win/tie/loss remains a
separate comparison metric and does not silently convert a broken candidate to
a pass.

When a frozen comparison is requalified by the deterministic request audit, the
publisher preserves the original pairwise outcome and adds a digest-bound
`request_identifiability.json` artifact to every case. It also exposes the audit
status and issue counts as primitive metrics and records the following two
cohorts in `benchmark_run.json.metadata.comparison`:

- `outcomes_for_cad_agent`: the unchanged raw comparison over every task.
- `request_identifiability.detected_conflict_free_outcomes_for_cad_agent`: only
  tasks for which the prompt audit returned `not_flagged`.

The second cohort is a reporting stratum, not a replacement score. A false
informational `request_identifiability` signal does not change case status.
Requalification is accepted only when all tasks are audited by one bound
classifier, no provider call or reference geometry was used, every aggregate is
derivable from case evidence, and the source comparison remains immutable.
`not_flagged` means no supported conflict was detected; it is not proof that the
prompt fully determines every shape-changing quantity.

All local artifact paths in a Geometry bundle or score are POSIX-style,
canonical, and relative to `bundle/`. Empty, `.`, and `..` segments, absolute
paths, URI schemes, and backslashes are invalid. External URI and data-image
references that the dashboard does not map to run files are not integrity
artifacts.
Every declared Geometry artifact, render image, and camera record has a lowercase
SHA-256 digest. A render record is accepted as exact OVRTX evidence only when it
also records the renderer, quality, mode, active AOV, sensor updates, dimensions,
camera, and camera digest. Elapsed render time is preserved when the renderer
reports it. `renders[view]`, when present, must identify the same image as
`geometry.render_evidence[view]`.

`status`, `geometry.outcome`, `handoff_ready`, and `geometry.findings` describe
workflow and handoff state. They do not determine the benchmark verdict.
Arbitrary primitive metrics are displayed without adding per-run UI fields.

## Artifact Integrity

Before a Geometry run becomes discoverable, the dashboard artifact server
establishes the real run root and takes bounded byte snapshots of every bound
regular file.
The bound set covers `benchmark_run.json`, `bundle/run.json`, optional
`score/suite_result.json`, optional `evaluation/material_evaluation.json`, and
every local file reachable through standard bundle, score, or Geometry fields.
This includes reviews, source/output USD, references, renders, artifact links,
local trace artifacts, score artifacts/thumbnails, declared Geometry artifacts,
and render image/camera evidence. `local_artifacts.run_dir` is not a file link
and is excluded.

Every path component must be a real directory or regular file under the verified
run root. Missing files, non-regular files, digest mismatches, symlinks (including
symlinks that remain inside the run), and escapes invalidate the entire run.
Files present in the run but absent from the consumer-derived bound set are not
served.

The generated index entry carries a SHA-256 snapshot token plus the verified,
canonical declaration set:

```json
{
  "artifact_version": "sha256:3333333333333333333333333333333333333333333333333333333333333333",
  "geometry_integrity": {
    "schema_version": "content-agent-benchmark-geometry-integrity.v1",
    "artifacts": [
      {
        "path": "benchmark_run.json",
        "sha256": "1111111111111111111111111111111111111111111111111111111111111111"
      },
      {
        "path": "bundle/assets/fixture_valve/renders/front.png",
        "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      },
      {
        "path": "bundle/run.json",
        "sha256": "2222222222222222222222222222222222222222222222222222222222222222"
      }
    ]
  }
}
```

Integrity paths are canonical and relative to the run root. The browser requires
the attestation to cover exactly the authoritative and consumer-reachable paths,
and requires Geometry-declared digests to agree with it. The local Vite artifact
server rejects decoded URLs with empty, `.`, or `..` segments and rejects paths
outside the verified set. `artifact_version` is the SHA-256 digest of the sorted
path-and-digest declaration set. Every Geometry artifact request, including
`HEAD`, must carry that exact token as the single `artifact_version` query
parameter. The server revalidates the run and rejects stale, absent, or duplicate
tokens with `409`, preventing a browser that loaded index revision A from mixing
it with artifact revision B. It serves accepted bytes from the same snapshot used
for hashing. Descriptor reads allocate exactly the pre-checked file size and
reject short reads, growth, or metadata mutation during capture. The snapshot
is limited to 256 MiB per file and 512 MiB per run; non-Geometry benchmark JSON
used for index classification is independently limited to 64 MiB.
Missing or malformed tokens are rejected before a run is scanned. Concurrent
requests for the same run share one in-flight verification, while process-wide
verification concurrency and the pending queue are bounded. Concurrent index
requests also share one in-flight scan. On the local Vite
server, Geometry serving requires Linux procfs: every opened file descriptor is
resolved before and after reading and must identify the expected regular file
under the verified run root. This closes parent-directory replacement races that
path-only `realpath` and `O_NOFOLLOW` checks cannot close.

Static and object-storage mirrors must generate the same index field from
verified bytes, enforce the same canonical run-relative allowlist and snapshot
token, and keep published run objects immutable.

## Authoritative Score

`score/suite_result.json` is the only source for case and aggregate benchmark
status. Each case uses the existing `pass`, `warn`, `fail`, `error`, or `skipped`
status and carries deterministic signals, metrics, artifacts, and thumbnails.
The dashboard turns failed signals into findings. A bundle asset absent from a
partial score remains explicitly unscored; an expected score case absent from
the bundle remains visible as a score-only case.

Conditional, rejected, unsupported, not-evaluated, and downstream-owned states
belong in the bundle extension or signal detail. Publishers must not coerce them
to pass or fail without an approved deterministic scoring rule.

## Publication Rules

1. Read evidence from caller-selected roots and stage it under a new run ID.
2. Copy only approved artifacts. Do not publish credentials, private prompts,
   workstation paths, or restricted source assets; publish approved digests and
   access references instead.
3. Validate JSON schemas, identity agreement, duplicate asset/case/view IDs,
   canonical relative paths, complete consumer-path coverage, regular-file
   existence, media types, and SHA-256 values against the staged file bytes
   before making the run discoverable. Do not publish a partial Geometry run
   when any bound artifact fails validation.
4. Write artifacts first and publish the complete run directory atomically.
   Refinement creates a new run ID; it never rewrites prior scored evidence.
5. Keep failed, rejected, and unscored runs. The dashboard is an evidence
   browser, not a mechanism for hiding unsuccessful results.

The local Vite server polls the filesystem index to discover complete runs. It
does not execute, resume, stream, or control benchmarks.
