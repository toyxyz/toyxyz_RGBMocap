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
    # Each attempt owns its archive. A failed download or a changed pin never
    # leaves reusable extraction state, and the previous executable stays put
    # until the new one passes the pin check.
    $stage = Join-Path $resources ('uv-stage-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $stage | Out-Null
    try {
        $zip = Join-Path $stage 'uv.zip'
        $extract = Join-Path $stage 'files'
        $url = 'https://releases.astral.sh/github/uv/releases/download/' + $lock.uv_version + '/uv-x86_64-pc-windows-msvc.zip'
        Write-Host ('Downloading the verified uv bootstrap ' + $lock.uv_version + '...')
        Invoke-WebRequest -Uri $url -OutFile $zip
        Expand-Archive -LiteralPath $zip -DestinationPath $extract
        $candidate = Get-ChildItem -LiteralPath $extract -Filter uv.exe -File -Recurse | Select-Object -First 1
        if (!$candidate -or (Get-Sha256 $candidate.FullName) -ne $lock.uv_sha256) {
            throw 'The downloaded uv executable does not match the pinned SHA-256. Retry installation.'
        }
        if (Test-Path -LiteralPath $uv -PathType Leaf) {
            [System.IO.File]::Replace($candidate.FullName, $uv, (Join-Path $stage 'previous.exe'))
        } else {
            [System.IO.File]::Move($candidate.FullName, $uv)
        }
    }
    finally {
        $stageItem = Get-Item -LiteralPath $stage -ErrorAction SilentlyContinue
        $expectedParent = (Get-Item -LiteralPath $resources).FullName
        if ($stageItem -and $stageItem.PSIsContainer -and
            -not ($stageItem.Attributes -band [IO.FileAttributes]::ReparsePoint) -and
            $stageItem.Parent.FullName -eq $expectedParent -and $stageItem.Name -match '^uv-stage-[0-9a-f]{32}$') {
            Remove-Item -LiteralPath $stageItem.FullName -Recurse -Force
        }
    }
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
