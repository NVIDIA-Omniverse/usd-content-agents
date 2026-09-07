$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}
$destination = Join-Path $repoRoot ".data/examples/scene-material-pass"
$revision = "0ed0dfbc539c9de99289771bd6848effe3ef5779"
$baseUrl = "https://media.githubusercontent.com/media/NVIDIA/simready-foundation/$revision"
$rawBaseUrl = "https://raw.githubusercontent.com/NVIDIA/simready-foundation/$revision"
$manifest = Join-Path $PSScriptRoot "assets.sha256"
$scene = Join-Path $destination "mini_workcell.usda"

$null = New-Item -ItemType Directory -Force -Path $destination
Remove-Item -LiteralPath $scene -Force -ErrorAction SilentlyContinue
foreach ($line in Get-Content -LiteralPath $manifest) {
    if (-not $line.Trim()) {
        continue
    }
    $expected, $relative = $line.Trim() -split "\s+", 2
    $target = Join-Path $destination $relative
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -eq $expected) {
            continue
        }
    }
    $sourcePath = $relative.Substring("simready-foundation/".Length).Replace("\", "/")
    $url = "$baseUrl/$sourcePath"
    if ($relative.EndsWith(".mdl") -or $relative.EndsWith(".txt")) {
        $url = "$rawBaseUrl/$sourcePath"
    }
    $partial = "$target.part"
    $null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target)
    try {
        Invoke-WebRequest -Uri $url -OutFile $partial
        $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) {
            throw "SimReady asset checksum mismatch: $relative"
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
    } finally {
        if (Test-Path -LiteralPath $partial) {
            Remove-Item -LiteralPath $partial -Force
        }
    }
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "mini_workcell.usda") `
    -Destination $scene -Force
Write-Output $scene
