$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path vendor,data,runs,web/public,research/provenance/licenses | Out-Null
$taskVendors = @(
    @{Name='flycube'; Url='https://github.com/lntegrals/flycube-public.git'; Revision='c14b3adcbccbd15e3036bbc502af1a77bb50fe92'},
    @{Name='unitree_rl_gym'; Url='https://github.com/unitreerobotics/unitree_rl_gym.git'; Revision='276801e46c5d433564f24658bac64f254b7d2d4b'}
)
foreach ($taskVendor in $taskVendors) {
    $taskPath = Join-Path 'vendor' $taskVendor.Name
    if (-not (Test-Path -LiteralPath $taskPath)) {
        git clone $taskVendor.Url $taskPath
        if ($LASTEXITCODE -ne 0) { throw 'Vendor download failed.' }
        git -C $taskPath checkout --detach $taskVendor.Revision
        if ($LASTEXITCODE -ne 0) { throw 'Vendor version selection failed.' }
    }
    $taskRevision = git -C $taskPath rev-parse HEAD
    if ($taskRevision -ne $taskVendor.Revision) { throw "Unexpected vendor revision: $taskPath" }
}
uv venv --python 3.12 .venv --allow-existing
if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed.' }
uv pip sync --python .venv/Scripts/python.exe requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --quiet
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
npm ci --no-audit --no-fund
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
if (-not (Test-Path -LiteralPath 'web/public/avatar.vrm')) {
    $taskProvenance = Get-Content -Raw -LiteralPath 'research/provenance/provenance.json' | ConvertFrom-Json
    Invoke-WebRequest -Uri $taskProvenance.avatar.url -OutFile 'web/public/avatar.vrm'
}
$taskExpected = (Get-Content -Raw -LiteralPath 'research/provenance/provenance.json' | ConvertFrom-Json).avatar.sha256
if ((Get-FileHash -Algorithm SHA256 -LiteralPath 'web/public/avatar.vrm').Hash.ToLower() -ne $taskExpected) { throw 'Avatar hash mismatch.' }
$taskYumiMeta = (Get-Content -Raw -LiteralPath 'research/provenance/provenance.json' | ConvertFrom-Json).avatar_yumi
if (Test-Path -LiteralPath $taskYumiMeta.file) {
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $taskYumiMeta.file).Hash.ToLower() -ne $taskYumiMeta.sha256) { throw 'Yumi VRM hash mismatch.' }
} else {
    Write-Host "Optional Yumi VRM missing ($($taskYumiMeta.file)); the page falls back to the default avatar. See research/avatar/YUMI.md."
}
.venv/Scripts/python.exe -m app.prepare
if ($LASTEXITCODE -ne 0) { throw 'Data preparation failed.' }
if (-not (Test-Path -LiteralPath 'runs/best.npz')) {
    .venv/Scripts/python.exe -m app.train
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
}
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
.venv/Scripts/python.exe -m app.check
if ($LASTEXITCODE -ne 0) { throw 'Validation failed.' }
Write-Host 'Ready. Run .\start.ps1 -Body g1 (or -Body yumi; the page can also switch bodies).'
