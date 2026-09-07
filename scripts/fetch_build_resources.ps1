# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Fetch the native-Windows Scene Optimizer Core build resources.
# Override the release asset with SO_CORE_URL only when SO_CORE_SHA256 is also
# set to the trusted SHA-256 of that archive.

[CmdletBinding()]
param(
    [string]$Url = $env:SO_CORE_URL,
    [string]$Sha256 = $env:SO_CORE_SHA256,
    [string]$BuildResources = $env:SO_CORE_BUILD_RESOURCES,
    [switch]$PrintUrlOnly
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($BuildResources)) {
    $BuildResources = Join-Path $RepoRoot ".build-resources"
}

$Architecture = $env:SO_CORE_ARCH
if ([string]::IsNullOrWhiteSpace($Architecture)) {
    $Architecture = $env:PROCESSOR_ARCHITECTURE
}
if ($Architecture -notin @("AMD64", "amd64", "x86_64")) {
    throw "Scene Optimizer Core supports native Windows x86_64 only; detected: $Architecture"
}

$Platform = "windows-x86_64"
$DefaultUrl = "https://github.com/NVIDIA-Omniverse/usd-optimize/releases/download/v1.0.3/scene_optimizer_core_usd_25.11_py_3.12%401.0.3.1-0-3.506.5ccdcb0b.gl.$Platform.release.zip"
$DefaultSha256 = "35a1e0992dd3c95ec56d93e282ecf6d47e635453cbac8ea243407cddda8fe253"
$UsingDefaultUrl = [string]::IsNullOrWhiteSpace($Url)
if ($UsingDefaultUrl) {
    $Url = $DefaultUrl
}

if ($PrintUrlOnly -or $env:SO_CORE_PRINT_URL_ONLY -in @("1", "true")) {
    Write-Output $Url
    return
}

if ([string]::IsNullOrWhiteSpace($Sha256)) {
    if (-not $UsingDefaultUrl) {
        throw "SO_CORE_SHA256 or -Sha256 is required when SO_CORE_URL or -Url overrides the release asset."
    }
    $Sha256 = $DefaultSha256
}
$TrustedSha256 = $Sha256.Trim().ToLowerInvariant()
if ($TrustedSha256 -notmatch "^[0-9a-f]{64}$") {
    throw "Scene Optimizer Core SHA-256 must contain exactly 64 hexadecimal characters."
}

function Get-StringSha256 {
    param([Parameter(Mandatory = $true)][string]$Value)

    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $Bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
        $Hash = $Hasher.ComputeHash($Bytes)
        return ([System.BitConverter]::ToString($Hash)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $Hasher.Dispose()
    }
}

function Test-ExpectedLayout {
    param([Parameter(Mandatory = $true)][string]$Path)

    foreach ($Subdirectory in @("python", "lib", "extraLibs", "usdpy")) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $Subdirectory) -PathType Container)) {
            return $false
        }
    }
    return $true
}

$UrlSha256 = Get-StringSha256 $Url
$PackageDir = Join-Path $BuildResources "scene_optimizer_core"
$MarkerPath = Join-Path $PackageDir ".so_core_platform"

function Test-InstalledPackageMatches {
    if (-not (Test-ExpectedLayout $PackageDir)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $MarkerPath -PathType Leaf)) {
        return $false
    }
    $Marker = Get-Content -LiteralPath $MarkerPath
    return (
        ($Marker -contains "platform=$Platform") -and
        ($Marker -contains "url_sha256=$UrlSha256") -and
        ($Marker -contains "archive_sha256=$TrustedSha256")
    )
}

if (Test-InstalledPackageMatches) {
    Write-Host "scene_optimizer_core already unpacked at $PackageDir for $Platform - skipping fetch"
    return
}

New-Item -ItemType Directory -Force -Path $BuildResources | Out-Null
$Nonce = [System.Guid]::NewGuid().ToString("N")
$TempDir = Join-Path $BuildResources "scene_optimizer_core.tmp.$Nonce"
$UnpackDir = Join-Path $TempDir "scene_optimizer_core"
$ZipPath = Join-Path $TempDir "scene_optimizer_core.zip"
$BackupDir = $null
$InstallStarted = $false
$OwnsPackageDir = $false
$InstallLock = $null
$InstallLockPath = Join-Path $BuildResources "scene_optimizer_core.install.lock"

New-Item -ItemType Directory -Path $UnpackDir | Out-Null
try {
    Write-Host "Fetching Scene Optimizer Core from:"
    Write-Host "  $Url"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $ZipPath
    $ActualSha256 = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $TrustedSha256) {
        throw "Scene Optimizer Core archive SHA-256 mismatch: expected $TrustedSha256, got $ActualSha256."
    }
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $UnpackDir
    Remove-Item -LiteralPath $ZipPath -Force

    if (-not (Test-ExpectedLayout $UnpackDir)) {
        throw "Unpacked Scene Optimizer Core package is missing an expected directory."
    }
    @(
        "platform=$Platform"
        "url_sha256=$UrlSha256"
        "archive_sha256=$TrustedSha256"
    ) | Set-Content -LiteralPath (Join-Path $UnpackDir ".so_core_platform") -Encoding ascii

    try {
        $InstallLock = [System.IO.File]::Open(
            $InstallLockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another Scene Optimizer Core installation is in progress; retry after it completes."
    }

    if (Test-InstalledPackageMatches) {
        Write-Host "scene_optimizer_core was installed by another setup process - skipping install"
        return
    }

    if (Test-Path -LiteralPath $PackageDir) {
        $BackupDir = Join-Path $BuildResources "scene_optimizer_core.backup.$Nonce"
        Move-Item -LiteralPath $PackageDir -Destination $BackupDir
    }
    $InstallStarted = $true

    $Installed = $false
    for ($Attempt = 1; $Attempt -le 5; $Attempt++) {
        try {
            if (Test-Path -LiteralPath $PackageDir) {
                throw "Scene Optimizer Core destination exists before install attempt $Attempt."
            }
            $OwnsPackageDir = $true
            Move-Item -LiteralPath $UnpackDir -Destination $PackageDir -ErrorAction Stop
            if (-not (Test-ExpectedLayout $PackageDir)) {
                throw "Installed Scene Optimizer Core package has an invalid layout."
            }
            $Installed = $true
            break
        }
        catch {
            if ($OwnsPackageDir -and (Test-Path -LiteralPath $PackageDir)) {
                Remove-Item -LiteralPath $PackageDir -Recurse -Force -ErrorAction SilentlyContinue
            }
            if (-not (Test-Path -LiteralPath $PackageDir)) {
                $OwnsPackageDir = $false
            }
            if (Test-Path -LiteralPath $PackageDir) {
                throw "Cannot retry Scene Optimizer Core installation while the partial destination remains."
            }
            if ($Attempt -eq 5) {
                throw
            }
            Write-Warning "Install attempt $Attempt failed; retrying."
            Start-Sleep -Seconds $Attempt
        }
    }
    if (-not $Installed) {
        throw "Failed to install Scene Optimizer Core."
    }

    if ($BackupDir -and (Test-Path -LiteralPath $BackupDir)) {
        Remove-Item -LiteralPath $BackupDir -Recurse -Force
        $BackupDir = $null
    }
    Write-Host "Installed Scene Optimizer Core into $PackageDir"
}
catch {
    $InstallError = $_
    if ($InstallStarted -and $OwnsPackageDir -and (Test-Path -LiteralPath $PackageDir)) {
        Remove-Item -LiteralPath $PackageDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $PackageDir)) {
        $OwnsPackageDir = $false
    }
    if ($BackupDir -and (Test-Path -LiteralPath $BackupDir)) {
        if (-not (Test-Path -LiteralPath $PackageDir)) {
            Move-Item -LiteralPath $BackupDir -Destination $PackageDir
            $BackupDir = $null
        }
    }
    throw $InstallError
}
finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force
    }
    if ($InstallLock) {
        $InstallLock.Dispose()
        Remove-Item -LiteralPath $InstallLockPath -Force -ErrorAction SilentlyContinue
    }
    if ($BackupDir -and (Test-Path -LiteralPath $BackupDir)) {
        Write-Warning "Previous Scene Optimizer Core package is preserved at $BackupDir"
    }
}
