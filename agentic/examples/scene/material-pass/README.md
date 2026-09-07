# Scene Material Pass

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash commands inside WSL2. PowerShell commands below are retained only
for development diagnostics and future qualification.

This example composes three recognizable assets from NVIDIA's public
[`simready-foundation`](https://github.com/NVIDIA/simready-foundation)
repository: a tool workbench, an electrician's toolbox, and a sledgehammer. The
workflow decomposes the scene, assigns materials to each asset, and collects the
validated bindings back onto the original scene topology.

The download is pinned to a Git revision and verifies every file against its
SHA-256 digest. It fetches about 25 MiB instead of the full repository. The USD
assets and textures are licensed under Apache-2.0, and the bundled OmniPBR MDL
modules include BSD-3-Clause notices. The fetch includes the upstream Apache
license; retain it and the MDL notices when redistributing the assets.

Fetch the assets once:

```bash
agentic/examples/scene/material-pass/fetch.sh
```

```powershell
.\agentic\examples\scene\material-pass\fetch.ps1
```

Then run the material pass:

```bash
agentic/examples/scene/material-pass/run.sh \
  --output-dir ../runs/scene-mini-workcell
```

```powershell
.\agentic\examples\scene\material-pass\run.ps1 `
  --output-dir ../runs/scene-mini-workcell
```

The final composed scene is written under
`03-collection/composition/composed_scene.usda` in the selected output
directory.
