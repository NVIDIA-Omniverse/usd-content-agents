$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

Push-Location (Join-Path $repoRoot "agentic")
try {
    & content-workflow-cli materials assign `
        --usd ../apps/material_agent/data/examples/ladder/sources/usd/ladder.usd `
        --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_1.jpeg `
        --reference-image ../apps/material_agent/data/examples/ladder/sources/images/ladder_reference_2.jpeg `
        --materials-yaml ../apps/material_agent/data/materials/material_libs_default/materials.yaml `
        --additional-instructions-file examples/materials/basic-assignment/material_guidance.md `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
