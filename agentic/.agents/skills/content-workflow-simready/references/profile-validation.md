# SimReady Profile Validation

Run formal SimReady Foundation profile validation on the latest staged USD
after conversion and requested content authoring workflows have produced a
meaningful artifact.

Formal profile validation is owned by `content_agent_workflows.simready`, not
by a scene backend. Run
`content-workflow-cli simready validate-profile` automatically after
dependency preflight against the staged asset and requested profile. The
wrapper invokes the external SimReady Foundation validator and writes the
normalized report. Do not prompt merely because usd-cli has no SimReady
command.

Validation reports include the selected profile, Foundation provenance,
command, feature results, requirement counts, issue counts, issues, ignored
issues, warnings, errors, rerun reasons, next step, and a canonical
`asset_dependency_manifest`. The manifest binds the root layer and every
resolved local USD composition or asset dependency by path, size, and SHA-256.
For USDZ, the archive SHA-256 binds the complete package.

Policy:

- Missing Foundation runtime or specs is `BLOCKED`, not a profile failure.
- Unresolved or remote dependencies are `BLOCKED`. A dependency-closure change
  during validation invalidates the run as `ERROR`; reusing evidence after any
  dependency changes blocks conformance.
- Failed profile requirements after a meaningful USD exists are diagnostic by
  default and should be recorded as conditional workflow status.
- Use `--strict` only when the caller wants failed validation to become a
  process failure.
- Treat `RB.MB.001` as non-blocking when topology inspection shows the asset is
  effectively a single mesh component or one `GeomSubset` component. Preserve
  the ignored issue in the report.

When validation fails with repairable requirements, hand the report to
`simready-conform-profile` and then rerun validation on the newest staged USD.
