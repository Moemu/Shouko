param(
    [ValidateSet('Status','Preview','Train','Sync')][string]$Action = 'Status',
    [ValidateRange(60,7200)][int]$Seconds = 1800,
    [string]$CloudHost = '',
    [string]$CloudPort = '',
    [string]$CloudRoot = '/root/autodl-tmp/neuromechfly'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Cloud target is machine-specific: pass -CloudHost/-CloudPort, or save them once in
# runs/cloud/target.json (git-ignored) as {"host": "user@host", "port": "12345"}.
$targetFile = 'runs/cloud/target.json'
if (-not $CloudHost -or -not $CloudPort) {
    if (Test-Path -LiteralPath $targetFile) {
        $saved = Get-Content -LiteralPath $targetFile -Raw | ConvertFrom-Json
        if (-not $CloudHost) { $CloudHost = $saved.host }
        if (-not $CloudPort) { $CloudPort = $saved.port }
    }
}
if (-not $CloudHost -or -not $CloudPort) {
    throw "Missing cloud target. Pass -CloudHost user@host -CloudPort 12345, or write them once to $targetFile (git-ignored)."
}
if ($Action -eq 'Sync') {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'UI build failed' }
    tar -czf runs/cloud/code.tgz app web/dist docs/CLOUD.md
    if ($LASTEXITCODE -ne 0) { throw 'Code archive failed' }
    scp -q -P $CloudPort runs/cloud/code.tgz "${CloudHost}:${CloudRoot}/code.tgz"
    if ($LASTEXITCODE -ne 0) { throw 'Code upload failed' }
    ssh -o BatchMode=yes -p $CloudPort $CloudHost "cd $CloudRoot && tar -xzf code.tgz"
} elseif ($Action -eq 'Train') {
    ssh -o BatchMode=yes -p $CloudPort $CloudHost "cd $CloudRoot && .venv/bin/python -m app.launch_cloud train --seconds $Seconds"
} elseif ($Action -eq 'Preview') {
    ssh -o BatchMode=yes -p $CloudPort $CloudHost "cd $CloudRoot && .venv/bin/python -m app.launch_cloud preview"
    if (-not (Get-NetTCPConnection -LocalPort 8742 -State Listen -ErrorAction SilentlyContinue)) {
        $cloudTunnel = Start-Process -FilePath ssh -ArgumentList @('-N','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=30','-L','127.0.0.1:8742:127.0.0.1:8742','-p',$CloudPort,$CloudHost) -WindowStyle Hidden -PassThru
        $cloudTunnel.Id | Set-Content runs/cloud/tunnel.pid
    }
    Write-Output 'Cloud studio: http://127.0.0.1:8742'
} else {
    ssh -o BatchMode=yes -p $CloudPort $CloudHost "nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader; tail -5 $CloudRoot/runs/cloud/train.log"
}
