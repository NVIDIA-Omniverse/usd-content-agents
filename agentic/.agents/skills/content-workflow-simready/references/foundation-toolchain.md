# SimReady Foundation Toolchain

SimReady Foundation is an external source of truth. The workflow adapter should
resolve a checkout, spec root, validator executable, and dedicated validation
environment before running conformance or validation.

Resolution order:

1. explicit CLI arguments such as `--foundation-root`, `--foundation-spec-root`,
   and `--venv`;
2. `SIMREADY_FOUNDATION_ROOT`, `SIMREADY_FOUNDATION_SPEC_ROOT`, and
   `CONTENT_WORKFLOW_SIMREADY_VENV`;
3. managed cache paths under `CONTENT_WORKFLOW_SIMREADY_CACHE_DIR` or the user
   cache directory.

The Foundation spec root normally contains:

```text
nv_core/sr_specs/docs/capabilities
nv_core/sr_specs/docs/features
nv_core/sr_specs/docs/profiles/profiles.toml
```

Do not fall back to local profile presets when Foundation specs or the
validator runtime are unavailable. Report `BLOCKED` with diagnostics.

The validation runtime is isolated from the main workflow environment because
`simready-validate`, `omniverse-asset-validator`, `usd-core`, and related
packages can conflict with other OpenUSD and Omniverse dependencies.

On Linux ARM64 with Python < 3.13, the managed validation runtime must use
`usd-exchange>=2.3,<3` as the active `pxr` provider because public `usd-core`
does not publish compatible wheels there. The adapter installs Foundation
requirements with `usd-core` excluded and adds `usd-exchange` explicitly. Set
`CONTENT_WORKFLOW_SIMREADY_USD_PROVIDER=usd-core` to force the standard
Foundation dependency path, or `CONTENT_WORKFLOW_SIMREADY_USD_PROVIDER=usd-exchange`
to force the workaround on another platform.
