# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Host setup for content-workflow-cli on Windows PowerShell.
#
# Native Windows can run content-workflow-cli against an existing Workbench URL. Local
# OvRTX Workbench rendering requires a Linux NVIDIA GPU host.

param(
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Test-CommandAvailable {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

Test-CommandAvailable uv
Test-CommandAvailable node
Test-CommandAvailable npm

Push-Location $RepoRoot
try {
    $VenvDir = Join-Path $RepoRoot ".venv"
    if ((Test-Path $VenvDir) -and -not $RecreateVenv) {
        Write-Host "Reusing existing .venv. Pass -RecreateVenv to rebuild it with Python 3.12."
    }
    elseif (Test-Path $VenvDir) {
        Remove-Item -Recurse -Force $VenvDir
        uv venv --python=3.12
    }
    else {
        uv venv --python=3.12
    }
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    uv pip install --python $Python -e (Join-Path $RepoRoot "agentic\packages\content_workflow_cli")
    npm ci --prefix (Join-Path $RepoRoot "agentic\packages\content_workflow_cli")
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "content-workflow-cli host setup complete."
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Verify Codex auth, using ChatGPT/OAuth if that is your normal Codex login:"
Write-Host "  content-workflow-cli auth login"
Write-Host ""
Write-Host "For headless hosts:"
Write-Host "  content-workflow-cli auth login --device-code"
Write-Host ""
Write-Host "Run content-workflow-cli against an existing Workbench endpoint with --workbench-url."
Write-Host "Local Workbench rendering requires a Linux NVIDIA GPU host."
