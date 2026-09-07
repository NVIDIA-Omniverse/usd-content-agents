# Basic Mesh Segmentation

Native Windows execution is unsupported in the 0.6 release. On a Windows host,
run the Bash command inside WSL2. The PowerShell launcher below is retained only
for development diagnostics and future qualification.

`fused_cart.usda` contains one mesh with three disconnected components: a body
and two wheels. It is deliberately small so the example demonstrates the
segmentation workflow rather than asset setup.

From the repository root:

```bash
agentic/examples/mesh-segmentation/basic/run.sh \
  --output-dir agentic/runs/mesh-segmentation-fused-cart
```

```powershell
agentic/examples/mesh-segmentation/basic/run.ps1 `
  --output-dir agentic/runs/mesh-segmentation-fused-cart
```
