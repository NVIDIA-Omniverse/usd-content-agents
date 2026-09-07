$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}
$sourceAsset = Join-Path $repoRoot ".data/examples/toycar/ToyCar.glb"
if (-not (Test-Path -LiteralPath $sourceAsset -PathType Leaf)) {
    throw "Missing $sourceAsset. Run agentic/examples/cad-to-simready/toycar/fetch.ps1 first."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli cad-to-simready run $sourceAsset `
        --asset-id toycar `
        --output-dir ../.local-runs/content-workflow-cli/cad-to-simready-toycar `
        --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
