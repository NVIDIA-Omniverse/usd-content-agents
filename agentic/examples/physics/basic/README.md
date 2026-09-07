# Basic Physics Workflow

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash command inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

This example reuses the packaged Lightbulb01 asset covered by the Physics Agent
end-to-end tests.

```bash
agentic/examples/physics/basic/run.sh
```

```powershell
agentic/examples/physics/basic/run.ps1
```

For development-only native Windows diagnostics, preflight provisions the
pinned OvPhysX runtime in an isolated venv before using the simulation path:

```powershell
content-workflow-cli preflight physics-runtime
agentic/examples/physics/basic/run.ps1
```

`agentic/examples/physics/basic/run.ps1 --no-simulation` retains
schema-authoring artifacts but returns a conditional, non-passing verdict. It
does not satisfy required runtime or visual evidence.

Pass another `--output-dir` when re-running the example.
