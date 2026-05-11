$ErrorActionPreference = "Stop"

$AppPort = 8000
$NgrokDomain = "cabbage-regroup-outright.ngrok-free.dev"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectRoot "log"
$PidDir = Join-Path $ProjectRoot ".run"
$BackendPidFile = Join-Path $PidDir "backend.pid"
$NgrokPidFile = Join-Path $PidDir "ngrok.pid"
$BackendLog = Join-Path $LogDir "backend.log"
$BackendErrLog = Join-Path $LogDir "backend.err.log"
$NgrokLog = Join-Path $LogDir "ngrok.log"
$NgrokErrLog = Join-Path $LogDir "ngrok.err.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $PidDir | Out-Null
if (-not (Test-Path $BackendLog)) { New-Item -ItemType File -Path $BackendLog | Out-Null }
if (-not (Test-Path $BackendErrLog)) { New-Item -ItemType File -Path $BackendErrLog | Out-Null }
if (-not (Test-Path $NgrokLog)) { New-Item -ItemType File -Path $NgrokLog | Out-Null }
if (-not (Test-Path $NgrokErrLog)) { New-Item -ItemType File -Path $NgrokErrLog | Out-Null }

function Test-BackendUp {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$AppPort/docs" -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch {
        return $false
    }
}

function Save-Pid($Path, $PidValue) {
    Set-Content -Path $Path -Value $PidValue -NoNewline
}

function Get-NgrokVersion {
    $line = (& ngrok version | Select-Object -First 1)
    if ($line -match "(\d+\.\d+\.\d+)") {
        return [Version]$Matches[1]
    }
    throw "Could not parse ngrok version output: $line"
}

function Test-NgrokTunnelUp {
    param([string]$ExpectedDomain)
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2
        if (-not $resp.tunnels) { return $false }
        foreach ($t in $resp.tunnels) {
            if ($t.public_url -match [Regex]::Escape($ExpectedDomain)) { return $true }
        }
        return $false
    } catch {
        return $false
    }
}

if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) {
    throw "ngrok not found in PATH."
}
$ngrokVersion = Get-NgrokVersion
$minimumNgrokVersion = [Version]"3.20.0"
if ($ngrokVersion -lt $minimumNgrokVersion) {
    throw "ngrok version $ngrokVersion is too old. Please update to >= $minimumNgrokVersion (`winget upgrade --id Ngrok.Ngrok -e`)."
}

if (Test-BackendUp) {
    Write-Host "[WARN] Backend already running on port $AppPort."
} else {
    Write-Host "[INFO] Starting backend on port $AppPort..."
    $VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $BackendProc = Start-Process -FilePath $VenvPython -ArgumentList "-m uvicorn backend.main:app --host 0.0.0.0 --port $AppPort" -WorkingDirectory $ProjectRoot -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErrLog -PassThru
    } else {
        $BackendProc = Start-Process -FilePath "python" -ArgumentList "-m uvicorn backend.main:app --host 0.0.0.0 --port $AppPort" -WorkingDirectory $ProjectRoot -RedirectStandardOutput $BackendLog -RedirectStandardError $BackendErrLog -PassThru
    }
    Save-Pid $BackendPidFile $BackendProc.Id
}

Write-Host "[INFO] Waiting for backend readiness..."
$ready = $false
for ($i = 0; $i -lt 45; $i++) {
    if (Test-BackendUp) { $ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    throw "Backend did not start. Check $BackendLog"
}
Write-Host "[OK] Backend is reachable at http://localhost:$AppPort"

$ngrokRunning = $false
if (Test-Path $NgrokPidFile) {
    $existingNgrokPid = Get-Content $NgrokPidFile -Raw
    if ($existingNgrokPid) {
        try {
            Get-Process -Id ([int]$existingNgrokPid) -ErrorAction Stop | Out-Null
            $ngrokRunning = $true
        } catch {}
    }
}

if ($ngrokRunning) {
    Write-Host "[WARN] ngrok is already running from previous run."
} else {
    Write-Host "[INFO] Starting ngrok with domain $NgrokDomain..."
    $NgrokProc = Start-Process -FilePath "ngrok" -ArgumentList "http $AppPort --domain=$NgrokDomain --log=stdout" -WorkingDirectory $ProjectRoot -RedirectStandardOutput $NgrokLog -RedirectStandardError $NgrokErrLog -PassThru
    Save-Pid $NgrokPidFile $NgrokProc.Id
}

Write-Host "[INFO] Waiting for ngrok tunnel readiness..."
$ngrokReady = $false
for ($i = 0; $i -lt 20; $i++) {
    if (Test-NgrokTunnelUp -ExpectedDomain $NgrokDomain) { $ngrokReady = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $ngrokReady) {
    throw "ngrok tunnel did not become ready for $NgrokDomain. Check $NgrokLog and $NgrokErrLog"
}
Write-Host "[OK] ngrok tunnel is active at https://$NgrokDomain"

Write-Host ""
Write-Host "====================================================="
Write-Host "  Service startup complete"
Write-Host "====================================================="
Write-Host "  Backend (local):    http://localhost:$AppPort"
Write-Host "  Backend (public):   https://$NgrokDomain"
Write-Host "  Docs (local):       http://localhost:$AppPort/docs"
Write-Host "  Docs (public):      https://$NgrokDomain/docs"
Write-Host "  Live logs below. Press Ctrl+C to stop tail."
Write-Host "====================================================="

Get-Content -Path $BackendLog, $BackendErrLog, $NgrokLog, $NgrokErrLog -Tail 30 -Wait
