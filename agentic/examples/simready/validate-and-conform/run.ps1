$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

$asset = Join-Path $repoRoot "apps/usd_cli/sample_assets/simready/Cube/cube.usda"
$outputDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    $env:CONTENT_WORKFLOW_OUTPUT_DIR
} else {
    Join-Path $repoRoot ".local-runs/content-workflow-cli/simready-cube"
}
$null = New-Item -ItemType Directory -Force -Path $outputDir
$report = Join-Path $outputDir "validation.json"
$conformed = Join-Path $outputDir "conformed"

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli simready validate-profile $asset `
        --profile Package `
        --report $report `
        --strict
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & content-workflow-cli simready conform-profile $asset `
        --output-dir $conformed `
        --profile Package `
        --validation-report $report `
        --strict `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
