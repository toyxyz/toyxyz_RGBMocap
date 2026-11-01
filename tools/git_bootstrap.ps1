$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$setup = Join-Path $root '.setup'
$resources = Join-Path $setup 'bootstrap-git'
New-Item -ItemType Directory -Force -Path $resources | Out-Null
$lock = Get-Content -LiteralPath (Join-Path $root 'installer/runtime-lock.json') -Encoding UTF8 -Raw | ConvertFrom-Json
$uv = Join-Path $resources 'uv.exe'
function Get-Sha256([string]$path) {
    $stream = [System.IO.File]::OpenRead($path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try { return ([System.BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
    finally { $hasher.Dispose(); $stream.Dispose() }
}
if (!(Test-Path -LiteralPath $uv) -or (Get-Sha256 $uv) -ne $lock.uv_sha256) {
    $zip = Join-Path $resources 'uv-download.zip'
    $extract = Join-Path $resources 'uv-extracted'
    $url = 'https://releases.astral.sh/github/uv/releases/download/' + $lock.uv_version + '/uv-x86_64-pc-windows-msvc.zip'
    if (!(Test-Path -LiteralPath $zip)) {
        Write-Host 'Downloading the verified uv bootstrap...'
        Invoke-WebRequest -Uri $url -OutFile $zip
    }
    if (!(Test-Path -LiteralPath $extract)) { Expand-Archive -LiteralPath $zip -DestinationPath $extract -Force }
    $candidate = Get-ChildItem -LiteralPath $extract -Filter uv.exe -File -Recurse | Select-Object -First 1
    if (!$candidate -or (Get-Sha256 $candidate.FullName) -ne $lock.uv_sha256) { throw 'The uv bootstrap hash does not match the pinned version. Remove the downloaded bootstrap archive and retry.' }
    Copy-Item -LiteralPath $candidate.FullName -Destination $uv -Force
}
$pythonDir = Join-Path $setup 'python'
$env:UV_PYTHON_INSTALL_DIR = $pythonDir
$env:UV_CACHE_DIR = Join-Path $setup 'uv-cache'
$env:UV_PYTHON_NO_REGISTRY = '1'
$env:UV_NO_CONFIG = '1'
$env:PYTHONUTF8 = '1'
& $uv python install 3.10.22 --install-dir $pythonDir --no-bin --no-registry --no-config --cache-dir $env:UV_CACHE_DIR
if ($LASTEXITCODE -ne 0) { throw 'Private Python setup failed.' }
$python = Join-Path $pythonDir 'cpython-3.10.22-windows-x86_64-none/python.exe'
& $python -I -X utf8 -B (Join-Path $root 'tools/git_managed_install.py') @args
exit $LASTEXITCODE
