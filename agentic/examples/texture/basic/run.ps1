$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

$gripPrim = "/RootNode/Geometry/M_AluminumStepLadder_B01_Plastic2"
$requestedAppearance = "smooth high-visibility safety orange rubberized plastic " +
    "with an even single-color matte finish"
$outputDir = if ($env:CONTENT_WORKFLOW_OUTPUT_DIR) {
    if ([System.IO.Path]::IsPathRooted($env:CONTENT_WORKFLOW_OUTPUT_DIR)) {
        [System.IO.Path]::GetFullPath($env:CONTENT_WORKFLOW_OUTPUT_DIR)
    } else {
        [System.IO.Path]::GetFullPath(
            (Join-Path $repoRoot $env:CONTENT_WORKFLOW_OUTPUT_DIR)
        )
    }
} else {
    Join-Path $repoRoot "runs/texture-ladder"
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
        & content-workflow-cli texture run `
            --usd ../apps/texture_agent/data/examples/ladder/sources/usd/ladder_uv_ready.usd `
            --prompt $requestedAppearance `
            --prim-path $gripPrim `
            --unit-action "$gripPrim=generate" `
            --unit-appearance "$gripPrim=$requestedAppearance" `
            --output-dir $outputDir `
            --runner codex `
            @args
        $commandExitCode = $LASTEXITCODE
    } finally {
        if ($hadNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    $handoff = Join-Path $outputDir "texture_companion_generation_handoff.json"
    if ($commandExitCode -eq 3 -and (Test-Path -LiteralPath $handoff -PathType Leaf)) {
        Write-Warning (
            "Texture paused for the outer coding-agent companion. Read $handoff, " +
            "write the exact requested result manifest, then resume with: " +
            "content-workflow-cli texture resume --run-dir `"$outputDir`""
        )
    }
} finally {
    Pop-Location
}
exit $commandExitCode
