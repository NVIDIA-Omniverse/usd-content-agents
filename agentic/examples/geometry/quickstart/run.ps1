# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}

Push-Location $repoRoot
try {
    & content-workflow-cli geometry run `
        agentic/examples/geometry/quickstart/smoke_bracket.usda `
        --output-dir agentic/runs/geometry-quickstart-001 `
        @args
    $commandExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $commandExitCode
