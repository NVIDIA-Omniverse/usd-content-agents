$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli physics apply `
        --usd ../apps/physics_agent/data/examples/Lightbulb01/light_bulb_01.usda `
        --output-dir ../.local-runs/content-workflow-cli/physics-lightbulb `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
