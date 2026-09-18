param(
    [ValidateRange(1,65535)][int]$Port = 8740
)
$ErrorActionPreference = 'Stop'
$taskListeners = Get-NetTCPConnection -LocalAddress '127.0.0.1' -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($taskListener in $taskListeners) {
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskListener.OwningProcess)"
    $taskParent = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskProcess.ParentProcessId)"
    $taskOwned = $taskProcess.CommandLine -like "*$PSScriptRoot*" -or $taskParent.CommandLine -like "*$PSScriptRoot*"
    if ($taskProcess.CommandLine -match 'uvicorn\s+app\.(studio_server|server|cloud_server):app' -and $taskOwned) {
        Stop-Process -Id $taskListener.OwningProcess
        Write-Host 'Flybody stopped.'
    } else {
        throw "Port $Port belongs to another process; it was not stopped."
    }
}
