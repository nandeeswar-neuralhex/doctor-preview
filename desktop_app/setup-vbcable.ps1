# ============================================================
# VB-Cable Post-Install Setup Script for Windows
# ============================================================
# Run this AFTER manually installing VB-Cable driver and rebooting.
# Admin NOT required. Internet required (first run only).
#
# Open PowerShell and run:
#   powershell -ExecutionPolicy Bypass -File .\setup-vbcable.ps1
#
# Or if you prefer to allow scripts permanently:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#   .\setup-vbcable.ps1
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===== VB-Cable Post-Install Setup =====" -ForegroundColor Cyan
Write-Host ""

# ── 1. Install AudioDeviceCmdlets module (if not already installed) ──
Write-Host "[1/6] Installing AudioDeviceCmdlets module..." -ForegroundColor Yellow

$nuget = Get-PackageProvider -Name NuGet -ErrorAction SilentlyContinue
if (-not $nuget -or $nuget.Version -lt [version]"2.8.5.201") {
    Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force | Out-Null
    Write-Host "  NuGet provider installed."
}

$mod = Get-Module -ListAvailable -Name AudioDeviceCmdlets
if (-not $mod) {
    Install-Module -Name AudioDeviceCmdlets -Force -Scope CurrentUser
    Write-Host "  AudioDeviceCmdlets module installed."
} else {
    Write-Host "  AudioDeviceCmdlets already installed, skipping."
}

Import-Module AudioDeviceCmdlets -Force

# ── 2. Verify VB-Cable is present ───────────────────────────────────
Write-Host ""
Write-Host "[2/6] Checking for VB-Cable..." -ForegroundColor Yellow

$allDevices = Get-AudioDevice -List
$cableInput = $allDevices | Where-Object { $_.Type -eq 'Playback' -and $_.Name -match 'CABLE Input' }
$cableOutput = $allDevices | Where-Object { $_.Type -eq 'Recording' -and $_.Name -match 'CABLE Output' }

if (-not $cableInput -or -not $cableOutput) {
    Write-Host ""
    Write-Host "ERROR: VB-Cable not detected!" -ForegroundColor Red
    Write-Host "Make sure you:" -ForegroundColor Red
    Write-Host "  1. Downloaded from https://vb-audio.com/Cable/" -ForegroundColor Red
    Write-Host "  2. Right-clicked VBCABLE_Setup_x64.exe -> Run as administrator" -ForegroundColor Red
    Write-Host "  3. Rebooted after installing" -ForegroundColor Red
    Write-Host ""
    Write-Host "All detected devices:" -ForegroundColor Yellow
    $allDevices | Format-Table Index, Type, Name -AutoSize
    exit 1
}

Write-Host "  Found: $($cableInput.Name)" -ForegroundColor Green
Write-Host "  Found: $($cableOutput.Name)" -ForegroundColor Green

# ── 3. Set real speakers as default output ───────────────────────────
Write-Host ""
Write-Host "[3/6] Setting real speakers as default output..." -ForegroundColor Yellow

$speaker = $allDevices | Where-Object { $_.Type -eq 'Playback' -and $_.Name -match 'Realtek' }

if (-not $speaker) {
    Write-Host "  WARNING: No Realtek device found. Listing playback devices:" -ForegroundColor Red
    $allDevices | Where-Object { $_.Type -eq 'Playback' } | Format-Table Index, Name -AutoSize
    $idx = Read-Host "  Enter the Index number of your real speaker/headphone device"
    $speaker = $allDevices | Where-Object { $_.Index -eq [int]$idx }
}

Set-AudioDevice -Index $speaker.Index | Out-Null
Set-AudioDevice -Index $speaker.Index -CommunicationOnly | Out-Null
Write-Host "  Default output -> $($speaker.Name)" -ForegroundColor Green

# ── 4. Set real microphone as default input ──────────────────────────
Write-Host ""
Write-Host "[4/6] Setting real microphone as default input..." -ForegroundColor Yellow

$mic = $allDevices | Where-Object { $_.Type -eq 'Recording' -and $_.Name -match 'Microphone Array' }

if (-not $mic) {
    # Fallback: try any recording device that isn't CABLE Output
    $mic = $allDevices | Where-Object { $_.Type -eq 'Recording' -and $_.Name -notmatch 'CABLE' } | Select-Object -First 1
}

if (-not $mic) {
    Write-Host "  WARNING: No real microphone found. Listing recording devices:" -ForegroundColor Red
    $allDevices | Where-Object { $_.Type -eq 'Recording' } | Format-Table Index, Name -AutoSize
    $idx = Read-Host "  Enter the Index number of your real microphone device"
    $mic = $allDevices | Where-Object { $_.Index -eq [int]$idx }
}

Set-AudioDevice -Index $mic.Index | Out-Null
Set-AudioDevice -Index $mic.Index -CommunicationOnly | Out-Null
Write-Host "  Default input -> $($mic.Name)" -ForegroundColor Green

# ── 5. Set volume levels ─────────────────────────────────────────────
Write-Host ""
Write-Host "[5/6] Setting volume levels..." -ForegroundColor Yellow

Set-AudioDevice -PlaybackVolume 80
Set-AudioDevice -RecordingVolume 90
Write-Host "  Speaker volume: 80%"
Write-Host "  Mic volume: 90%"

# ── 6. Verify and print final status ─────────────────────────────────
Write-Host ""
Write-Host "[6/6] Final verification..." -ForegroundColor Yellow
Write-Host ""
Write-Host "========== FINAL STATUS ==========" -ForegroundColor Cyan

Write-Host ""
Write-Host "--- Default Output ---" -ForegroundColor White
$out = Get-AudioDevice -Playback
Write-Host "  Device : $($out.Name)"
Write-Host "  Volume : $(Get-AudioDevice -PlaybackVolume)%"
Write-Host "  Muted  : $(Get-AudioDevice -PlaybackMute)"

Write-Host ""
Write-Host "--- Default Input ---" -ForegroundColor White
$inp = Get-AudioDevice -Recording
Write-Host "  Device : $($inp.Name)"
Write-Host "  Volume : $(Get-AudioDevice -RecordingVolume)%"
Write-Host "  Muted  : $(Get-AudioDevice -RecordingMute)"

Write-Host ""
Write-Host "--- All Devices ---" -ForegroundColor White
Get-AudioDevice -List | Format-Table Index, Default, DefaultCommunication, Type, Name -AutoSize

# ── Test beep ─────────────────────────────────────────────────────────
Write-Host ""
[console]::beep(1000, 500)
Write-Host "Did you hear a beep? If yes, speakers are working!" -ForegroundColor Green

# ── Reminder ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "========== MANUAL STEP REMAINING ==========" -ForegroundColor Yellow
Write-Host ""
Write-Host "In Google Meet -> Settings (gear icon) -> Audio:" -ForegroundColor White
Write-Host '  Microphone -> "CABLE Output (VB-Audio Virtual Cable)"' -ForegroundColor White
Write-Host '  Speaker    -> "Speaker / Headphone (Realtek Audio)"' -ForegroundColor White
Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
