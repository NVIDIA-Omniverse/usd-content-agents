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
    Join-Path $repoRoot "runs/articulation-mini-drawer"
}

Push-Location $repoRoot
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
        & content-workflow-cli preflight articulation-authoring-platform
        $preflightExitCode = $LASTEXITCODE
        if ($preflightExitCode -ne 0) {
            $commandExitCode = $preflightExitCode
        } else {
            & content-workflow-cli asset run `
                --usd (Join-Path $repoRoot "agentic/examples/articulation/drawer/drawer.usda") `
                --output-dir $outputDir `
                --prompt "Use the sole long-running coordinator to author exactly one prismatic joint that moves /MiniCabinet/Drawer along the Y axis relative to /MiniCabinet/Frame. Use deterministic scene evidence by default; do not select an external articulation proposal provider. Do not author masses or colliders, and preserve all non-target topology." `
                --required-leaf articulation.author.v1 `
                --required-leaf articulation.evidence.v1 `
                --required-leaf articulation.preparation-publisher.v1 `
                --required-leaf articulation.publish.v1 `
                --required-leaf articulation.review.v1 `
                --required-leaf validation.canonical-ovrtx-evidence.v1 `
                --required-terminal-leaf articulation.publish.v1 `
                --required-terminal-leaf validation.canonical-ovrtx-evidence.v1 `
                --required-leaf-dependency articulation.author.v1=articulation.preparation-publisher.v1 `
                --required-leaf-dependency articulation.evidence.v1=articulation.author.v1 `
                --required-leaf-dependency articulation.review.v1=articulation.evidence.v1 `
                --required-leaf-dependency articulation.publish.v1=articulation.review.v1 `
                --required-leaf-dependency validation.canonical-ovrtx-evidence.v1=articulation.author.v1 `
                --exact-leaf-scope `
                --runner codex `
                @args
            $commandExitCode = $LASTEXITCODE
        }
    } finally {
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }
} finally {
    Pop-Location
}
exit $commandExitCode
