# Geometry Stage

Use the frozen `request.geometry` policy and exact `input_asset`; do not invent
optimization, repair, segmentation, render, runtime, or SimReady settings. When
the predecessor is external authoring, bind its immutable source manifest,
selected representation, source-bundle identity, provider receipt, and
parameters. Do not rediscover or regenerate source intent.

After sealing the Geometry plan and calling `begin-stage`, run:

```bash
content-workflow-asset-state execute-geometry \
  --run-state "$RUN_STATE"
```

The command writes `geometry_stage_result.json` in the current attempt and
`domain-run/geometry_workflow_result.json`. Inspect the typed result, durable
`.usdc`, manifest, validation evidence, evidence bundle, optimization metadata,
shared USD report, and required OVRTX report and images.

Exit `0` means the typed result is admissible or conditional; exit `1` means it
is rejected but preserved for review/refinement; exit `2` is a command or
contract error.

Bind `result_path`, `workflow_result.path`, and every unique binding under
`artifacts`. Set `output_asset_path` to `output_asset.path`. Acceptance requires
an exact match to the active run, plan, input, attempt, output, dependencies,
manifest, validation evidence, and requested OVRTX source/image digests. For an
externally authored source, the source-bundle and selected-representation
identity must survive in the Geometry manifest.

`handoff_ready=no` is blocked. Optimizer fallback may be accepted only as an
explicitly conditional handoff. A conditional handoff is admissible only when
every Geometry-owned gate passes and each remaining condition is assigned to a
later owner, such as Material, Physics/runtime, or publication provenance.

The accepted output must be binary Z-up, `metersPerUnit=1.0` `.usdc`. Its
complete dependency closure becomes the exact Articulation input. On failure,
record `refine`, `revisit`, or a typed stop decision; never continue with the
original source.
