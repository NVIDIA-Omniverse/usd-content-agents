$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

$outputDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    if ([System.IO.Path]::IsPathRooted($env:CONTENT_WORKFLOW_OUTPUT_DIR)) {
        [System.IO.Path]::GetFullPath($env:CONTENT_WORKFLOW_OUTPUT_DIR)
    } else {
        [System.IO.Path]::GetFullPath(
            (Join-Path $repoRoot $env:CONTENT_WORKFLOW_OUTPUT_DIR)
        )
    }
} else {
    Join-Path $repoRoot "runs/validation-smoke-bracket"
}
$model = if ($env:CONTENT_WORKFLOW_MODEL) {
    $env:CONTENT_WORKFLOW_MODEL
} else {
    "gpt-5.6-sol"
}
$renderBackend = if ($env:CONTENT_WORKFLOW_RENDER_BACKEND) {
    $env:CONTENT_WORKFLOW_RENDER_BACKEND.Trim().ToLowerInvariant()
} else {
    "ovrtx"
}
if ($renderBackend -notin @("ovrtx", "remote")) {
    throw "CONTENT_WORKFLOW_RENDER_BACKEND must be 'ovrtx' or 'remote'."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    $nativePreference = Get-Variable `
        -Name PSNativeCommandUseErrorActionPreference `
        -ErrorAction SilentlyContinue
    $hadNativePreference = $null -ne $nativePreference
    $previousNativePreference = if ($hadNativePreference) {
        $nativePreference.Value
    } else {
        $null
    }
    try {
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }
        & content-workflow-cli validate run `
            --usd (Join-Path $repoRoot "agentic/examples/geometry/quickstart/smoke_bracket.usda") `
            --task "Validate static USD integrity and current OVRTX rendering for release readiness. Do not modify the asset; report evidence, limitations, and recommended actions." `
            --output-dir $outputDir `
            --runner codex `
            --model $model `
            --render-backend $renderBackend `
            @args
        $commandExitCode = $LASTEXITCODE
    } finally {
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }
} finally {
    Pop-Location
}
exit $commandExitCode
