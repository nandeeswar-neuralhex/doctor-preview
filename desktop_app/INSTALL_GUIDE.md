# Doctor Preview — macOS Installation Guide

## For Developers (Building the App)

### Prerequisites

- Node.js (v18+)
- npm

### Build the macOS App

```bash
cd desktop_app
npm install
npm run build:mac
```

The DMG installer will be created at:

```
desktop_app/dist/Doctor Preview-1.0.0-arm64.dmg
```

---

## For Users (Installing the App)

### Step 1: Install Homebrew (if not installed)

Open **Terminal** (Spotlight → type "Terminal") and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

- Enter your **Mac password** when prompted (characters won't show — that's normal)
- Press **Return/Enter** when asked to continue
- After installation, run these 3 commands:

```bash
echo >> ~/.zprofile
echo 'eval "$(/opt/homebrew/bin/brew shellenv zsh)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
```

Verify Homebrew:

```bash
brew --version
```

### Step 2: Install BlackHole Audio Driver

```bash
brew install blackhole-2ch
```

### Step 3: Install Doctor Preview

1. Download `Doctor Preview-1.0.0-arm64.dmg`
2. **Before opening the DMG**, open Terminal and run:

```bash
xattr -d com.apple.quarantine ~/Downloads/Doctor\ Preview-1.0.0-arm64.dmg
```

3. Double-click the DMG to mount it
4. Drag **Doctor Preview** to the **Applications** folder
5. Open the app from Applications

---

## Troubleshooting

### "Doctor Preview is damaged and can't be opened"

This happens because the app is not code-signed. macOS quarantines all downloaded files.

#### Fix Option 1 — System Settings (Easiest)

1. Try opening the app (it will fail — that's OK)
2. Go to **Apple menu → System Settings → Privacy & Security**
3. Scroll down — you'll see: _"Doctor Preview" was blocked_
4. Click **"Open Anyway"**
5. Enter your password and confirm

#### Fix Option 2 — Terminal Commands

Open Terminal and run these commands one by one:

```bash
# Remove quarantine from the DMG
sudo xattr -r -d com.apple.quarantine ~/Downloads/Doctor\ Preview-1.0.0-arm64.dmg

# Eject any mounted disk image
hdiutil detach "/Volumes/Doctor Preview 1.0.0-arm64" 2>/dev/null

# Mount the DMG fresh
hdiutil attach ~/Downloads/Doctor\ Preview-1.0.0-arm64.dmg -nobrowse

# Copy to Applications
sudo cp -R "/Volumes/Doctor Preview 1.0.0-arm64/Doctor Preview.app" /Applications/

# Strip all extended attributes
sudo xattr -cr /Applications/Doctor\ Preview.app

# Open the app
open /Applications/Doctor\ Preview.app
```

#### Fix Option 3 — Temporarily Disable Gatekeeper (Last Resort)

```bash
# Disable Gatekeeper
sudo spctl --master-disable

# Open the app
open /Applications/Doctor\ Preview.app

# Re-enable Gatekeeper after app opens successfully
sudo spctl --master-enable
```

---

## Important Notes

- ⚠️ **Do NOT copy-paste commands from messaging apps** (WhatsApp, iMessage, etc.) — they add invisible characters that break Terminal. **Type commands manually**.
- The app is built for **Apple Silicon (M1/M2/M3/M4)**. If the other Mac is Intel, a universal build is needed.
- The quarantine fix is a **one-time step** per installation.
- When entering your password in Terminal, **no characters will appear** — just type and press Enter.

---

## System Requirements

| Requirement  | Details                      |
| ------------ | ---------------------------- |
| macOS        | 10.12 (Sierra) or later      |
| Architecture | Apple Silicon (arm64)        |
| Disk Space   | ~200 MB                      |
| Audio Driver | BlackHole 2ch (via Homebrew) |
