# Doctor Preview - Desktop App

Desktop application for doctors to preview post-surgery results on patients in real-time.

## Features

- 📸 Upload 1-10 target images (post-surgery previews)
- 🎥 Live webcam feed
- 👁️ Side-by-side view: Original vs. AI-processed
- 🔄 Real-time face swap at 24+ FPS
- 🌐 Connects to RunPod cloud service

## Tech Stack

- **Electron** - Cross-platform desktop app (Windows/Mac)
- **React** - UI framework
- **WebSocket** - Real-time streaming
- **Tailwind CSS** - Styling

## Quick Start

```bash
cd desktop_app
npm install
npm start
```

## Build for Distribution

```bash
# macOS
npm run build:mac

# Windows
npm run build:win
```
