# Public beta from this laptop, through a free Cloudflare quick tunnel.
#   powershell -ExecutionPolicy Bypass -File deploy\start_public_beta.ps1
# (or double-click "Start EngineLab beta" on the desktop)
# Runs the publishable package (deploy\space: public samples only, beta limits on), keeps the laptop
# awake while it runs, shows the https://....trycloudflare.com link, copies it and opens it.
# Closing the window or Ctrl+C stops the beta. The link changes every time this starts.
# After code changes, rebuild the package first:  python tools\build_space.py

param([int]$Port = 7860, [switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$space = Join-Path $root "deploy\space"
$cloudflared = Join-Path $env:USERPROFILE "tools\cloudflared\cloudflared.exe"
$port = $Port
$urlFile = Join-Path $PSScriptRoot "beta_link_$port.txt"
$Host.UI.RawUI.WindowTitle = "EngineLab beta"

function Show-Link($url, $note) {
    Write-Host ""
    Write-Host "  EngineLab beta is live$note" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "     $url" -ForegroundColor White -BackgroundColor DarkMagenta
    Write-Host ""
}

if (-not (Test-Path (Join-Path $space "run.py"))) { throw "deploy\space is missing - run: python tools\build_space.py" }
if (-not (Test-Path $cloudflared)) { throw "cloudflared.exe not found at $cloudflared" }

# Already running? Show the current link instead of starting a second copy.
$listen = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
if ($listen) {
    if (Test-Path $urlFile) {
        $url = (Get-Content $urlFile -Raw).Trim()
        Show-Link $url " (it was already running)"
        try { Set-Clipboard $url; Write-Host "  Link copied to the clipboard." } catch {}
    } else {
        Write-Host "EngineLab is already running on this laptop, but its link is unknown." -ForegroundColor Yellow
    }
    Write-Host ""
    $ans = Read-Host "  Type R to restart it with a new link, S to stop it, or press Enter to leave it running"
    if ($ans -notmatch '^[rRsS]') { return }
    foreach ($c in $listen) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }
    Get-CimInstance Win32_Process -Filter "Name='cloudflared.exe'" | Where-Object { $_.CommandLine -match ":$port" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Remove-Item $urlFile -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    if ($ans -match '^[sS]') { Write-Host "Beta stopped."; return }
}

# Keep the laptop from sleeping while the beta is up (released automatically when this window closes).
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$env:ENGINELAB_BETA = "1"
$env:HOST = "127.0.0.1"          # only reachable through the tunnel, not from the local network
$env:PORT = "$port"
$env:PYTHONUNBUFFERED = "1"
$log = Join-Path $env:TEMP "enginelab_beta_server_$port.log"
$tlog = Join-Path $env:TEMP "enginelab_beta_tunnel_$port.log"
Remove-Item $urlFile, $tlog -ErrorAction SilentlyContinue

Write-Host "Starting the EngineLab server..."
$server = Start-Process python -ArgumentList "run.py", "$port" -WorkingDirectory $space -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
$tunnel = $null
try {
    $ok = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Seconds 1
        if ($server.HasExited) { Get-Content "$log.err" -Tail 30; throw "The server stopped while starting." }
        try { Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/" -TimeoutSec 2 | Out-Null; $ok = $true; break } catch {}
    }
    if (-not $ok) { throw "The server did not answer on port $port." }

    Write-Host "Server is up. Opening the public link..."
    $tunnel = Start-Process $cloudflared -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port" `
        -WindowStyle Hidden -RedirectStandardError $tlog -RedirectStandardOutput "$tlog.out" -PassThru
    $url = $null
    for ($i = 0; $i -lt 90 -and -not $url; $i++) {
        Start-Sleep -Seconds 1
        if ($tunnel.HasExited) { Get-Content $tlog -Tail 20; throw "The tunnel stopped while starting (no internet?)." }
        if (Test-Path $tlog) {
            $m = Select-String -Path $tlog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { throw "Cloudflare did not give a link. Check the internet connection and try again." }
    Set-Content -Path $urlFile -Value $url -Encoding utf8

    # The new address can take a few seconds to become reachable.
    for ($i = 0; $i -lt 30; $i++) {
        try { Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 5 -UserAgent "Mozilla/5.0" | Out-Null; break } catch { Start-Sleep -Seconds 2 }
    }
    Show-Link $url ""
    try { Set-Clipboard $url; Write-Host "  The link is copied - paste it to your testers." } catch {}
    if (-not $NoBrowser) { Start-Process $url }
    Write-Host ""
    Write-Host "  Keep this window open while people test. Press Ctrl+C here to stop the beta" -ForegroundColor DarkGray
    Write-Host "  (or double-click the desktop icon again and choose S)." -ForegroundColor DarkGray

    while (-not $tunnel.HasExited -and -not $server.HasExited) { Start-Sleep -Seconds 2 }
    if ($server.HasExited) { Write-Host "The server stopped unexpectedly:" -ForegroundColor Red; Get-Content "$log.err" -Tail 20 }
    if ($tunnel.HasExited) { Write-Host "The tunnel stopped unexpectedly:" -ForegroundColor Red; Get-Content $tlog -Tail 10 }
}
finally {
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Remove-Item $urlFile -ErrorAction SilentlyContinue
    [void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000000")
    Write-Host "Beta stopped."
}
