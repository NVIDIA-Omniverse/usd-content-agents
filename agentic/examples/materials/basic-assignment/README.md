# Basic Material Assignment

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash command inside WSL2. The PowerShell launcher below is retained only
for development diagnostics and future qualification.

This example reuses the unbound ladder and two reference images already shipped
with Material Agent.

```bash
agentic/examples/materials/basic-assignment/run.sh \
  --output-dir agentic/runs/materials-ladder
```

Development-only native PowerShell launcher:

```powershell
agentic/examples/materials/basic-assignment/run.ps1 `
  --output-dir agentic/runs/materials-ladder
```
