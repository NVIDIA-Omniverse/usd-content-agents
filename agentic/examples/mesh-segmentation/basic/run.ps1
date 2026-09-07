$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli mesh-segmentation run `
        --asset examples/mesh-segmentation/basic/fused_cart.usda `
        --target-prim /FusedCart/Geometry `
        --target-semantic-part body `
        --target-semantic-part wheel `
        --allow-codex-configured-auth `
        --no-memory `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
