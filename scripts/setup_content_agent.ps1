# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development-only host setup helper for content-workflow-cli on Windows PowerShell.
#
# Native Windows execution and local OVRTX rendering are unsupported in the 0.6
# release. On a Windows host, run supported workflows inside WSL2; this helper
# remains available for development evidence and future qualification only.

param(
    [switch]$RecreateVenv,
    [switch]$SkipBuildResources,
    [switch]$WithoutChildRunners,
    [string]$UvExecutable,
    [string]$NodeExecutable,
    [string]$NpmExecutable
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path

function Resolve-RequiredCommand {
    param(
        [string]$Name,
        [string[]]$AdditionalCandidates = @()
    )
    foreach ($Candidate in $AdditionalCandidates) {
        if (
            -not [string]::IsNullOrWhiteSpace($Candidate) -and
            (Test-Path -LiteralPath $Candidate -PathType Leaf)
        ) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
    }
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) {
        return $Command.Source
    }
    throw "Required command not found: $Name"
}

function Assert-NodeAtLeast20 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$NodeExecutable
    )
    $Reported = (& $NodeExecutable --version 2>$null)
    $Major = 0
    if ($Reported -match '^v?(\d+)') {
        $Major = [int]$Matches[1]
    }
    if ($Major -lt 20) {
        $Found = if ($Reported) { $Reported } else { "unknown" }
        throw ("Node.js 20 or newer is required for the Codex and Claude child " +
               "runners (found: $Found). Install Node 20+, or rerun with " +
               "-WithoutChildRunners to set up direct Python workflows only.")
    }
}

$UvCommand = Resolve-RequiredCommand uv @($UvExecutable)
if (-not $WithoutChildRunners) {
    $NodeInstallDir = Join-Path $env:ProgramFiles "nodejs"
    $NodeCommand = Resolve-RequiredCommand node @(
        $NodeExecutable,
        (Join-Path $NodeInstallDir "node.exe")
    )
    Assert-NodeAtLeast20 $NodeCommand
    $NpmCommand = Resolve-RequiredCommand npm @(
        $NpmExecutable,
        (Join-Path $NodeInstallDir "npm.cmd")
    )
    # A newly installed machine-wide Node may not be visible to the current
    # PowerShell process yet. Keep npm and the generated Codex command shims
    # usable for setup verification and subsequent CLI calls in this shell.
    $PathEntries = @($env:PATH -split ";")
    if ($NodeInstallDir -notin $PathEntries) {
        $env:PATH = "$NodeInstallDir;$env:PATH"
    }
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArguments,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )
    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Push-Location $RepoRoot
$PreviousPwdEnvironment = [Environment]::GetEnvironmentVariable("PWD", "Process")
try {
    # The root pyproject intentionally uses file://${PWD} for first-party SDF
    # constraints in --no-sources release lanes. PowerShell's $PWD is not an
    # environment variable, so provide uv the file-URL path form for this
    # process while repository packages are resolved.
    $RepoUri = [Uri]::new($RepoRoot + [IO.Path]::DirectorySeparatorChar)
    $env:PWD = "/" + $RepoUri.AbsolutePath.TrimEnd("/")

    $VenvDir = Join-Path $RepoRoot ".venv"
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    if ((Test-Path $VenvDir) -and -not $RecreateVenv) {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
            throw (
                "Existing .venv has no Scripts\python.exe. " +
                "Rerun with -RecreateVenv to rebuild it."
            )
        }
        Write-Host "Reusing existing .venv. Pass -RecreateVenv to rebuild it with Python 3.12."
    }
    elseif (Test-Path $VenvDir) {
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
        Invoke-CheckedCommand $UvCommand @(
            "venv", $VenvDir, "--python", "3.12"
        ) "uv venv"
    }
    else {
        Invoke-CheckedCommand $UvCommand @(
            "venv", $VenvDir, "--python", "3.12"
        ) "uv venv"
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Virtual-environment Python was not found: $Python"
    }
    $MaterialAgentPackage = (Join-Path $RepoRoot "apps\material_agent") + "[all]"
    $UsdCliPackage = (Join-Path $RepoRoot "apps\usd_cli") + "[cli,server]"
    # uv 0.12.x reparses an absolute --overrides value containing spaces even
    # though PowerShell passes it as one argv item. Setup already runs from the
    # repository root, so keep this value relative.
    $UsdExchangeOverride = "apps\usd_cli\requirements\usd-exchange-override.txt"
    $ContentWorkflowPackage = Join-Path $RepoRoot "agentic\packages\content_workflow_cli"
    Invoke-CheckedCommand $UvCommand @(
        "pip", "install", "--python", $Python, "-e", $MaterialAgentPackage
    ) "material-agent installation"
    Invoke-CheckedCommand $UvCommand @(
        "pip", "install", "--python", $Python, "-e", $UsdCliPackage,
        "--overrides", $UsdExchangeOverride
    ) "usd-cli installation"
    Invoke-CheckedCommand $UvCommand @(
        "pip", "install", "--python", $Python, "-e", $ContentWorkflowPackage,
        "--overrides", $UsdExchangeOverride
    ) "content-workflow-cli installation"
    if (-not $WithoutChildRunners) {
        Invoke-CheckedCommand $NpmCommand @(
            "ci", "--prefix", $ContentWorkflowPackage
        ) "Node SDK installation"
    }
    else {
        Write-Host "Skipping Node SDK setup. Child-agent workflows remain unavailable."
    }
    if (-not $SkipBuildResources) {
        $BuildResourceScript = Join-Path $RepoRoot "scripts\fetch_build_resources.ps1"
        & $BuildResourceScript
    }
    else {
        Write-Host "Skipping Scene Optimizer build-resource fetch."
    }
}
finally {
    if ($null -eq $PreviousPwdEnvironment) {
        Remove-Item Env:PWD -ErrorAction SilentlyContinue
    }
    else {
        $env:PWD = $PreviousPwdEnvironment
    }
    Pop-Location
}

Write-Host ""
Write-Host "content-workflow-cli host setup complete."
Write-Host ""
Write-Host "Activate the environment:"
$ActivationScript = Join-Path $VenvDir "Scripts\Activate.ps1"
Write-Host ('  . "' + $ActivationScript + '"')
if (-not $WithoutChildRunners) {
    Write-Host ""
    Write-Host "Verify Codex auth, using ChatGPT/OAuth if that is your normal Codex login:"
    Write-Host "  content-workflow-cli auth login"
    Write-Host ""
    Write-Host "For headless hosts:"
    Write-Host "  content-workflow-cli auth login --device-code"
}
else {
    Write-Host ""
    Write-Host "This environment is configured for direct Python workflows only."
}
Write-Host ""
Write-Host "Native Windows workflow execution is unsupported in the 0.6 release."
Write-Host "This PowerShell helper is retained for development and future qualification only; run supported workflows inside WSL2."
Write-Host ""
Write-Host "material-agent, physics-agent, joint-agent, and texture-agent are installed."
Write-Host "Their standalone fixed-pipeline interfaces remain opt-in. The general setup"
Write-Host "does not install validation-agent. Use WSL2 with Warp for supported execution."
