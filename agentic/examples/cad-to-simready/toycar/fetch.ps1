$ErrorActionPreference = "Stop"

$repoRoot = (& git -C $PSScriptRoot rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0 -or -not $repoRoot) {
    throw "Unable to resolve the repository root."
}
$destination = Join-Path $repoRoot ".data/examples/toycar/ToyCar.glb"
$partial = "$destination.part"
$revision = "0e3a605bda7c758293ab58432f1d51a2a355d47a"
$expected = "01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c"
$url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/$revision/Models/ToyCar/glTF-Binary/ToyCar.glb"

$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination)
try {
    Invoke-WebRequest -Uri $url -OutFile $partial
    $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "ToyCar checksum mismatch: expected $expected, received $actual"
    }
    Move-Item -LiteralPath $partial -Destination $destination -Force
} finally {
    if (Test-Path -LiteralPath $partial) {
        Remove-Item -LiteralPath $partial -Force
    }
}
Write-Output $destination
