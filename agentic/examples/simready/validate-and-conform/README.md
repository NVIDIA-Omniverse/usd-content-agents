# SimReady Validate and Conform

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash command inside WSL2. The PowerShell launcher below is retained only
for development diagnostics and future qualification.

This example runs profile validation and conformance on the tiny SimReady Cube
package already shipped with usd-cli.

```bash
agentic/examples/simready/validate-and-conform/run.sh
```

```powershell
agentic/examples/simready/validate-and-conform/run.ps1
```

Set `CONTENT_WORKFLOW_OUTPUT_DIR` to use another output directory.
