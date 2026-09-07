# Asset Coordinator Control Reference

Read this reference before authoring a coordinator plan/review or invoking a
state transition. Both draft schemas reject unknown fields.

## Plan draft

```json
{
  "schema_version": "content-agent-workflows.asset-coordinator-plan-draft.v1",
  "stage": "material",
  "objective": "Produce reviewed material evidence.",
  "steps": [{
    "stage": "material",
    "objective": "Prepare and finalize the typed Material result.",
    "acceptance_evidence": ["coordinator_result.json and its bound evidence"],
    "may_revisit": true
  }],
  "evidence_paths": ["/absolute/path/to/inspected/evidence.json"],
  "revision_reason": "Initial Material plan from the frozen prompt and Joint handoff."
}
```

## Review draft

```json
{
  "schema_version": "content-agent-workflows.asset-coordinator-review-draft.v1",
  "stage": "material",
  "output_asset_path": "/absolute/path/to/material.usdz",
  "evidence_paths": ["/absolute/path/to/coordinator_result.json"],
  "findings": ["The result satisfies the plan acceptance evidence."],
  "decision": "accept",
  "target_stage": null,
  "decision_summary": "Accept the exact reviewed Material result.",
  "repair_scope": []
}
```

For `revisit`, set `target_stage` to the earlier stage and provide a non-empty
`repair_scope`. For `refine`, keep `target_stage` null and provide a non-empty
`repair_scope`. `accept` requires `output_asset_path`; other decisions may use
null when no output was produced.

## State commands

```bash
content-workflow-asset-state status --run-state RUN/asset_run.json
content-workflow-asset-state stage-dir --run-state RUN/asset_run.json --stage STAGE
content-workflow-asset-state record-plan --run-state RUN/asset_run.json \
  --plan-file PLAN_DRAFT.json
content-workflow-asset-state begin-stage --run-state RUN/asset_run.json --stage STAGE
content-workflow-asset-state record-evidence-review \
  --run-state RUN/asset_run.json --review-file REVIEW_DRAFT.json
content-workflow-asset-state require-review \
  --run-state RUN/asset_run.json --candidates FILE
content-workflow-asset-state complete-stage --run-state RUN/asset_run.json \
  --stage STAGE --output-asset ASSET --evidence FILE --summary TEXT
content-workflow-asset-state fail-stage \
  --run-state RUN/asset_run.json --stage STAGE --reason TEXT
content-workflow-asset-state cancel-stage \
  --run-state RUN/asset_run.json --stage STAGE --reason TEXT
content-workflow-asset-state build-report --run-state RUN/asset_run.json \
  --final-asset ASSET --validation-summary FILE --output FILE
content-workflow-asset-state validate-terminal --run-state RUN/asset_run.json
```

Repeat `--evidence` for every accepted artifact. Use `asset resume --recover
"reason"` only after the previous runner is confirmed gone and the concrete
cause is corrected.

## Failure dispositions

- `identity changed`: preserve the run for diagnosis. Never bless changed
  bytes; start a fresh run for changed user inputs.
- `needs_review`: inspect the candidates and evidence, then use `asset review`
  with one decision per required ID.
- Failed or cancelled stage: confirm the old process is gone, fix the cause,
  then use `asset resume --recover "reason"`.
- Never invoke `asset run`, `asset review`, or `asset resume` from the
  coordinator child. Record the recursive-lifecycle error and return.
- Conditional domain output: retain its evidence and record `refine`,
  `revisit`, or `stop_failed` with an explicit repair scope.
- Final USDZ rejection: regenerate through the converter and retain its report;
  never hand-edit the archive.
