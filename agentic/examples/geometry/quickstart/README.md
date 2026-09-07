# Geometry Quickstart Fixture

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash command inside WSL2. The PowerShell launcher below is retained only
for development diagnostics and future qualification.

`smoke_bracket.usda` is the small, meter-scale, Z-up input for the public Geometry
workflow quickstart. It tests source admission, durable USDC output, shared USD
validation, OVRTX evidence, manifest production, and CLI status handling without
introducing CAD conversion or model-generation variables.

From the repository root after setup:

```bash
content-workflow-cli geometry run \
  agentic/examples/geometry/quickstart/smoke_bracket.usda \
  --output-dir agentic/runs/geometry-quickstart-001
```

Development-only native PowerShell launcher:

```powershell
agentic/examples/geometry/quickstart/run.ps1
```

Use a fresh output directory for every run. See
[`../../../docs/geometry_quickstart.md`](../../../docs/geometry_quickstart.md) for
installation, OVRTX provisioning, exit codes, output interpretation, and
production entry points.
