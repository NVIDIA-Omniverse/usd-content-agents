# Coordinator Run Records

Use these exact JSON shapes. Both schemas reject unknown fields.

## Plan Draft

```json
{
  "schema_version": "content-agent-workflows.asset-coordinator-plan-draft.v1",
  "stage": "material",
  "objective": "Produce reviewed material evidence.",
  "steps": [
    {
      "stage": "material",
      "objective": "Prepare and finalize the typed Material result.",
      "acceptance_evidence": ["coordinator_result.json and its bound evidence"],
      "may_revisit": true
    }
  ],
  "evidence_paths": ["/absolute/path/to/inspected/evidence.json"],
  "revision_reason": "Initial Material plan from the frozen prompt and Articulation handoff."
}
```

## Review Draft

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
