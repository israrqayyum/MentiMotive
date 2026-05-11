$ErrorActionPreference = "SilentlyContinue"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidDir = Join-Path $ProjectRoot ".run"
$BackendPidFile = Join-Path $PidDir "backend.pid"
$NgrokPidFile = Join-Path $PidDir "ngrok.pid"
$BackendLog = Join-Path $ProjectRoot "log\backend.log"
$BackendErrLog = Join-Path $ProjectRoot "log\backend.err.log"
$NgrokLog = Join-Path $ProjectRoot "log\ngrok.log"
$NgrokErrLog = Join-Path $ProjectRoot "log\ngrok.err.log"

function Stop-ByPidFile {
    param(
        [string]$Name,
        [string]$PidFile
    )

    if (-not (Test-Path $PidFile)) {
        Write-Host "[INFO] No $Name pid file found."
        return
    }

    $rawPid = Get-Content $PidFile -Raw
    if (-not $rawPid) {
        Remove-Item $PidFile -Force
        Write-Host "[WARN] Empty pid file for $Name removed."
        return
    }

    $pidInt = 0
    [void][int]::TryParse($rawPid.Trim(), [ref]$pidInt)
    if ($pidInt -gt 0) {
        $proc = Get-Process -Id $pidInt
        if ($proc) {
            Stop-Process -Id $pidInt -Force
            Write-Host "[OK] Stopped $Name (PID $pidInt)."
        } else {
            Write-Host "[WARN] $Name process not running (PID $pidInt)."
        }
    } else {
        Write-Host "[WARN] Invalid pid in $PidFile."
    }

    Remove-Item $PidFile -Force
}

Stop-ByPidFile -Name "ngrok" -PidFile $NgrokPidFile
Stop-ByPidFile -Name "backend" -PidFile $BackendPidFile

Write-Host "[INFO] Stop routine complete."
Write-Host "[INFO] Recent backend logs:"
if (Test-Path $BackendLog) {
    Get-Content $BackendLog -Tail 20
} else {
    Write-Host "[INFO] No backend log file found."
}
if (Test-Path $BackendErrLog) {
    Write-Host "[INFO] Recent backend error logs:"
    Get-Content $BackendErrLog -Tail 20
}

Write-Host "[INFO] Recent ngrok logs:"
if (Test-Path $NgrokLog) {
    Get-Content $NgrokLog -Tail 20
} else {
    Write-Host "[INFO] No ngrok log file found."
}
if (Test-Path $NgrokErrLog) {
    Write-Host "[INFO] Recent ngrok error logs:"
    Get-Content $NgrokErrLog -Tail 20
}
