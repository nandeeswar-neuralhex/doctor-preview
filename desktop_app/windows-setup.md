# VB-Cable Windows Setup Guide (Full Automation)


## Overview
This sets up VB-Cable as a virtual audio pipe so the Doctor Preview app can route processed audio to Google Meet. The system speaker and mic stay untouched — only Google Meet uses the virtual cable.


## How It Works
```
Your real mic → WebRTC → Server (face swap + lip sync) → WebRTC back
                                                              ↓
                              Frontend setSinkId() → CABLE Input (only processed audio)
                                                              ↓
                                                        CABLE Output
                                                              ↓
                                          Google Meet picks up as "microphone" ✅


System audio (music, YouTube, etc.) → Realtek Speakers ✅ (never touches CABLE)
```


---


## Step 1: Download VB-Cable
Download from: https://vb-audio.com/Cable/
Extract to a folder (e.g., `C:\Users\<you>\Downloads\VBCABLE_Driver_Pack45`)


## Step 2: Install (requires manual Admin click)
Right-click `VBCABLE_Setup_x64.exe` → **Run as administrator** → Click **Install Driver**


> ⚠️ This step CANNOT be automated — Windows requires manual UAC confirmation for kernel audio drivers.


## Step 3: Reboot
```powershell
shutdown /r /t 10 /c "Rebooting for VB-Cable driver"
```


## Step 4: After Reboot — Run This PowerShell Script (Admin NOT required)


Open PowerShell and run the following commands **one by one** or paste the entire block:


```powershell
# ── 1. Install the AudioDeviceCmdlets module (one-time) ──────────────
Install-PackageProvider -Name NuGet -MinimumVersion 2.8.5.201 -Force
Install-Module -Name AudioDeviceCmdlets -Force -Scope CurrentUser


# ── 2. List all devices to verify VB-Cable is installed ──────────────
Get-AudioDevice -List | Format-Table Index, Default, DefaultCommunication, Type, Name -AutoSize


# ── 3. Find device indices ───────────────────────────────────────────
# Look at the output above and note the Index numbers for:
#   - "Speaker / Headphone (Realtek...)"  → e.g. Index 2
#   - "Microphone Array (Intel...)"       → e.g. Index 5
# Replace the numbers below if they differ on your machine.


# ── 4. Set Speaker as default output (both default + communications) ─
$speakerIndex = (Get-AudioDevice -List | Where-Object { $_.Type -eq 'Playback' -and $_.Name -match 'Realtek' }).Index
Set-AudioDevice -Index $speakerIndex
Set-AudioDevice -Index $speakerIndex -CommunicationOnly
Write-Host "✅ Speaker set as default output device"


# ── 5. Set real Microphone as default input (both default + communications) ─
$micIndex = (Get-AudioDevice -List | Where-Object { $_.Type -eq 'Recording' -and $_.Name -match 'Microphone Array' }).Index
Set-AudioDevice -Index $micIndex
Set-AudioDevice -Index $micIndex -CommunicationOnly
Write-Host "✅ Microphone Array set as default input device"


# ── 6. Set volume levels ─────────────────────────────────────────────
Set-AudioDevice -PlaybackVolume 80
Set-AudioDevice -RecordingVolume 90
Write-Host "✅ Speaker volume: 80%, Mic volume: 90%"


# ── 7. Verify everything ─────────────────────────────────────────────
Write-Host ""
Write-Host "========== FINAL STATUS =========="
Write-Host "--- Output ---"
$out = Get-AudioDevice -Playback
Write-Host "  Device: $($out.Name)"
Write-Host "  Default: $($out.Default)  CommDefault: $($out.DefaultCommunication)"
Write-Host "  Volume: $(Get-AudioDevice -PlaybackVolume)%  Muted: $(Get-AudioDevice -PlaybackMute)"
Write-Host ""
Write-Host "--- Input ---"
$inp = Get-AudioDevice -Recording
Write-Host "  Device: $($inp.Name)"
Write-Host "  Default: $($inp.Default)  CommDefault: $($inp.DefaultCommunication)"
Write-Host "  Volume: $(Get-AudioDevice -RecordingVolume)%  Muted: $(Get-AudioDevice -RecordingMute)"
Write-Host ""
Write-Host "--- All Devices ---"
Get-AudioDevice -List | Format-Table Index, Default, DefaultCommunication, Type, Name -AutoSize


# ── 8. Test beep ─────────────────────────────────────────────────────
[console]::beep(1000, 500)
Write-Host "🔊 Did you hear a beep? If yes, speakers are working!"
```


## Step 5: Google Meet Settings (manual)
In Google Meet → Settings (⚙️) → Audio:
- 🎤 **Microphone** → `CABLE Output (VB-Audio Virtual Cable)`
- 🔊 **Speaker** → `Speaker / Headphone (Realtek Audio)`


---


## Expected Final State


| Device | System Default | Comm Default | Used By |
|--------|---------------|-------------|---------|
| **Speaker / Headphone (Realtek)** | ✅ Yes | ✅ Yes | All system audio + Google Meet speaker |
| CABLE Input (VB-Audio) | ❌ No | ❌ No | Frontend code only (via setSinkId) |
| CABLE In 16ch | ❌ No | ❌ No | Not used |
| **Microphone Array (Intel)** | ✅ Yes | ✅ Yes | System default mic |
| CABLE Output (VB-Audio) | ❌ No | ❌ No | Google Meet mic only (manual selection) |


## Troubleshooting


### No audio from speakers
```powershell
# Check volume
Get-AudioDevice -PlaybackVolume
# Set to 80%
Set-AudioDevice -PlaybackVolume 80
# Check if muted
Get-AudioDevice -PlaybackMute
# Test
[console]::beep(1000, 500)
# Open volume mixer to check per-app levels
Start-Process sndvol.exe
```


### Other person hears my system audio
CABLE Input got set as default output. Fix:
```powershell
$i = (Get-AudioDevice -List | Where-Object { $_.Type -eq 'Playback' -and $_.Name -match 'Realtek' }).Index
Set-AudioDevice -Index $i
Set-AudioDevice -Index $i -CommunicationOnly
```


### Other person can't hear me at all
Check if frontend found VB-Cable. Open browser DevTools console, look for:
`[WebRTC-Audio] Remote audio routed to BlackHole: CABLE Input`
If you see "BlackHole/VB-Audio not found", the Electron app can't detect VB-Cable.


### Uninstall VB-Cable
Settings → Apps → search "VB-Audio Cable" → Uninstall → Reboot


### Uninstall Voicemeeter (if accidentally installed instead)
Settings → Apps → search "VB-Audio Voicemeeter" → Uninstall → Reboot
```powershell
# Verify it's gone (should show NO Voicemeeter entries)
Get-AudioDevice -List | Format-Table Name, Type -AutoSize
```



