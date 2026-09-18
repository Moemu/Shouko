param(
    [ValidateSet('g1','yumi')][string]$Body = 'g1',
    [ValidateSet('cuda','cpu')][string]$Device = 'cuda',
    [ValidateRange(1,65535)][int]$Port = 8740
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
try {
    $meta = Invoke-RestMethod "http://127.0.0.1:$Port/api/meta" -TimeoutSec 2
    if ($meta.body.robot -eq $Body) {
        Write-Host "Flybody is already running: http://127.0.0.1:$Port ($Body body)"
        return
    }
    Write-Host "Port $Port is serving the $($meta.body.robot) body. Switch in the page, or stop it with .\stop.ps1 -Port $Port first."
    return
} catch {}
$localPython = if ($Device -eq 'cuda') { "$taskRoot\.venv-gpu\Scripts\python.exe" } else { "$taskRoot\.venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $localPython)) { throw 'Missing Python environment. See research/guides/LOCAL.md.' }
# No checkpoint required: without best.pt the page still runs and can start training.
if (-not (Test-Path -LiteralPath "$taskRoot\web\dist\index.html")) {
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed.' }
}
$logDir = if ($Body -eq 'yumi') { 'runs/yumi' } else { 'runs/cloud' }
New-Item -ItemType Directory -Path "$taskRoot\$logDir" -Force | Out-Null
$previousDevice = $env:FLYBODY_DEVICE
$previousBody = $env:FLYBODY_BODY
try {
    $env:FLYBODY_DEVICE = $Device
    $env:FLYBODY_BODY = $Body
    $taskProcess = Start-Process -FilePath $localPython -ArgumentList '-m','uvicorn','app.studio_server:app','--host','127.0.0.1','--port',"$Port" -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput "$taskRoot\$logDir\server.log" -RedirectStandardError "$taskRoot\$logDir\server-error.log" -PassThru
    $taskProcess.Id | Set-Content -LiteralPath "$taskRoot\$logDir\server.pid"
} finally {
    $env:FLYBODY_DEVICE = $previousDevice
    $env:FLYBODY_BODY = $previousBody
}
Write-Host "Starting Flybody: http://127.0.0.1:$Port ($Body body)"
Write-Host "Startup log: $logDir\server-error.log"
