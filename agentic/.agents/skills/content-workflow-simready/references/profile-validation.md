# SimReady Profile Validation

Run formal SimReady Foundation profile validation on the latest staged USD
after conversion and requested content authoring workflows have produced a
meaningful artifact.

Validation command shape:

```bash
content-workflow-simready-validate-profile asset.usda \
  --profile Prop-Robotics-Neutral \
  --profile-version 1.0.0 \
  --report simready-profile.json
```

Validation reports include the selected profile, Foundation provenance,
command, feature results, requirement counts, issue counts, issues, ignored
issues, warnings, errors, rerun reasons, and next step.

Policy:

- Missing Foundation runtime or specs is `BLOCKED`, not a profile failure.
- Failed profile requirements after a meaningful USD exists are diagnostic by
  default and should be recorded as conditional workflow status.
- Use `--strict` only when the caller wants failed validation to become a
  process failure.
- Treat `RB.MB.001` as non-blocking when topology inspection shows the asset is
  effectively a single mesh component or one `GeomSubset` component. Preserve
  the ignored issue in the report.

When validation fails with repairable requirements, hand the report to
`simready-conform-profile` and then rerun validation on the newest staged USD.
