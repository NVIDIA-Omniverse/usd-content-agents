# Pinned IsaacLab External Refinement Example

> [!IMPORTANT]
> This is an experimental source-checkout walkthrough. It is not installed in
> the Physics Agent wheel and does not yet have a public real-GPU CI gate.

This example demonstrates the trusted-local Physics Agent external-runtime
contract. It refines an IsaacLab cube's restitution against a user-requested
bounce behavior. The customer simulator remains in its own checkout and
environment; its adapter reports a scalar objective and exact rollout recordings
while Physics Agent owns qualification, optimization, judging, and result
artifacts. The adapter calculates the one scalar objective used for candidate
selection; its other metrics are diagnostics only.

Read the public
[External Runtime Tuning and Local Refinement](../../docs/external_runtime_tuning.md)
guide before adapting this example to another repository.

## Setup

Install Physics Agent with its `tuning` extra in the World Understanding
environment that drives optimization. The IsaacLab runtime remains separate.

From the World Understanding checkout:

```bash
uv pip install -e "apps/physics_agent[tuning]"

export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=Y

python apps/physics_agent/examples/byor_isaaclab/bootstrap.py \
  --config-out /tmp/byor-isaaclab.json
```

The bootstrap clones IsaacLab at commit
`100ef39f128207cb536f87c77a7481fa416f6a7d`, creates its official
`env_isaaclab` uv environment, installs pinned Isaac Sim `6.0.1.0` with
its extension cache plus the core IsaacLab package, launches and closes Kit as
a smoke test, and writes an absolute-path runtime configuration. Existing dirty
checkouts are never modified. Use `--skip-install` when the pinned environment
is already provisioned, and use `--skip-smoke` only when GPU-backed Kit
validation is intentionally deferred.

The emitted config requests Isaac Sim Kit RTX qualification evidence as PNG
frames at 960x720 and 30 samples per second. Each Bayesian optimization
candidate records the exact cube pose trajectory as time-sampled USD without
paying per-trial image-rendering cost.

The EULA acceptance and optional privacy-consent variables are forwarded only
when explicitly set by the operator; the bootstrap does not accept NVIDIA
terms on the user's behalf.

## Qualify

The first invocation executes one nominal trial and stops:

```bash
physics-agent refine-external /tmp/byor-isaaclab.json \
  --output-dir /tmp/byor-isaaclab-run \
  --user-prompt "make the cube bounce to about 0.55 metres"
```

Review `/tmp/byor-isaaclab-run/qualification/qualification.json` and
`/tmp/byor-isaaclab-run/qualification/qualification_frames.json` plus the PNGs
under `/tmp/byor-isaaclab-run/qualification/qualification_frames/`, then rerun
the command with the exact printed digest. The digest binds the manifest,
declared image metadata, and every PNG byte, so replacing reviewed evidence
blocks optimization:

```bash
physics-agent refine-external /tmp/byor-isaaclab.json \
  --output-dir /tmp/byor-isaaclab-run \
  --user-prompt "make the cube bounce to about 0.55 metres" \
  --approve-qualification sha256:<digest>
```

Each iteration writes its active search, complete trial history, best parameters,
scalar objective result, exact winning recording, directly rendered PNGs, and
VLM verdict. The generated config also names `trajectory` in
`publish_artifacts`, so only the selected replica's trajectory is promoted to
the public winner bundle.
Failed customer trials remain in history but cannot become the best candidate.
Physics Agent renders the winning recording without rerunning IsaacLab.
The VLM consumes those PNGs directly; no MP4 encoding or frame resampling occurs.
This example authors the cube, ground, lighting, camera, and time-sampled poses
directly into each recording USD, so the recording has no external dependencies.
Custom adapters must likewise return a self-contained, single-file recording.
Physics Agent copies that file to `best_recording.usd`; it does not flatten USD
composition arcs or package referenced assets and textures.
The terminal iteration is published atomically to `final/` without simulation
or rendering:

```text
/tmp/byor-isaaclab-run/iter_N/best_params.json
/tmp/byor-isaaclab-run/iter_N/best_recording.usd
/tmp/byor-isaaclab-run/iter_N/render/frame_*.png
/tmp/byor-isaaclab-run/iter_N/judge.json
/tmp/byor-isaaclab-run/final/best_recording.usd
/tmp/byor-isaaclab-run/final/render/frame_*.png
/tmp/byor-isaaclab-run/final/outputs/trajectory/trajectory.json
/tmp/byor-isaaclab-run/final/{run_spec.json,history.jsonl,result.json,manifest.json}
```

Raw adapter requests, subprocess logs, and per-trial recordings remain in the
iteration directory and are not copied into `final/`.

The VLM therefore judges direct image evidence from an execution that actually
contributed to optimization.

`tune-external` remains available for one fixed scalar objective. Local
`refine-external` may change only active parameters and bounds. It cannot change
the adapter-owned objective. If task setup, adapter code, parameter exposure, or
objective calculation changes, the runtime receives a new fingerprint and must
be qualified again. The fingerprint also hashes the installed files in every
Isaac Sim Python distribution, so changing, editing, or upgrading
`env_isaaclab` invalidates the previous qualification. External tuning and
refinement are not exposed by Physics Agent Service.

## Trust Boundary

This feature executes a pre-provisioned local command. Use it only with a
trusted repository. It is not an arbitrary-repository upload mechanism and is
not exposed by service or public NVCF deployments.
