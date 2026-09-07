$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}
$sourceScene = Join-Path $repoRoot ".data/examples/scene-material-pass/mini_workcell.usda"
if (-not (Test-Path -LiteralPath $sourceScene -PathType Leaf)) {
    throw "Missing $sourceScene. Run agentic/examples/scene/material-pass/fetch.ps1 first."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli scene run `
        --usd $sourceScene `
        --task material `
        --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml `
        --additional-instructions-file examples/scene/material-pass/material_guidance.md `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
