# ToyCar CAD-to-SimReady

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash commands inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

ToyCar is a 5.4 MB CC0 GLB from the Khronos glTF Sample Assets repository. The
download is pinned to a commit and SHA-256.

```bash
agentic/examples/cad-to-simready/toycar/fetch.sh
agentic/examples/cad-to-simready/toycar/run.sh
```

```powershell
agentic/examples/cad-to-simready/toycar/fetch.ps1
agentic/examples/cad-to-simready/toycar/run.ps1
```

For development-only native Windows diagnostics, preflight can provision the
pinned OvPhysX runtime for the composed Physics stage:

```powershell
content-workflow-cli preflight physics-runtime
agentic/examples/cad-to-simready/toycar/run.ps1
```

The launcher stays intentionally small and delegates the pipeline to
`content-workflow-cli cad-to-simready run`.
