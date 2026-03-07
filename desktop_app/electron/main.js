const { app, BrowserWindow, ipcMain, globalShortcut } = require('electron');
const path = require('path');
const { exec } = require('child_process');
const http = require('http');
const fs = require('fs');

// In production (packaged DMG) serve the built files over localhost so Clerk
// sees a proper http:// origin instead of null (file://) which causes 401s.
let staticServer = null;
const PROD_PORT = 3474;

function startStaticServer(distPath) {
    return new Promise((resolve, reject) => {
        const MIME = {
            '.html': 'text/html',
            '.js': 'application/javascript',
            '.css': 'text/css',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
        };
        staticServer = http.createServer((req, res) => {
            const safePath = req.url.split('?')[0].replace(/\.\./g, '');
            let filePath = path.join(distPath, safePath === '/' ? 'index.html' : safePath);
            if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
                filePath = path.join(distPath, 'index.html'); // SPA fallback
            }
            const ext = path.extname(filePath).toLowerCase();
            const contentType = MIME[ext] || 'application/octet-stream';
            fs.readFile(filePath, (err, data) => {
                if (err) { res.writeHead(404); res.end('Not found'); return; }
                res.writeHead(200, { 'Content-Type': contentType });
                res.end(data);
            });
        });
        staticServer.listen(PROD_PORT, '127.0.0.1', () => resolve());
        staticServer.on('error', reject);
    });
}

// Prevent Chromium from throttling timers, canvas, and the renderer process
// when the window is minimized, hidden, or loses focus. Without these switches
// setTimeout intervals get clamped to ~1000ms (from ~41ms) and AudioContext
// gets suspended — both of which break the 24 FPS face-swap streaming loop.
app.commandLine.appendSwitch('disable-renderer-backgrounding');
app.commandLine.appendSwitch('disable-background-timer-throttling');
app.commandLine.appendSwitch('disable-backgrounding-occluded-windows');
app.commandLine.appendSwitch('disable-features', 'CalculateNativeWinOcclusion');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false,
            webSecurity: false,          // Allow cross-origin requests (Vite HMR + RunPod)
            backgroundThrottling: false, // Keep timers & canvas at full speed in background
            preload: path.join(__dirname, 'preload.js')
        },
        title: 'Doctor Preview',
        backgroundColor: '#1a1a1a'
    });

    // Load from Vite dev server in development, or built files in production
    const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

    if (isDev) {
        // Wait for Vite to be fully ready before loading
        const loadDevServer = () => {
            mainWindow.loadURL('http://localhost:3000').catch(() => {
                // Retry if Vite isn't ready yet
                setTimeout(loadDevServer, 1000);
            });
        };
        loadDevServer();
    } else {
        // Static server is started once in app.whenReady() — just load the URL
        mainWindow.loadURL(`http://localhost:${PROD_PORT}/`);
    }

    // Log renderer crashes and errors to terminal
    mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDesc) => {
        console.error(`[Electron] Page failed to load: ${errorCode} ${errorDesc}`);
    });
    mainWindow.webContents.on('render-process-gone', (event, details) => {
        console.error(`[Electron] Renderer crashed:`, details);
    });
    mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
        if (level >= 2) { // warnings and errors
            console.error(`[Renderer] ${message}`);
        }
    });

    // Open DevTools on demand: F12 or Cmd+Shift+I
    // Intercept Cmd+R / Cmd+Shift+R to log out instead of reloading
    mainWindow.webContents.on('before-input-event', (event, input) => {
        if (input.key === 'F12' ||
            (input.meta && input.shift && input.key.toLowerCase() === 'i')) {
            mainWindow.webContents.toggleDevTools();
        }
        // Cmd+R or Cmd+Shift+R → trigger logout instead of page refresh
        if (input.meta && (input.key.toLowerCase() === 'r') && input.type === 'keyDown') {
            event.preventDefault();
            mainWindow.webContents.send('trigger-logout');
        }
        // Cmd+H → go invisible (keep window alive for OBS) instead of native hide
        if (input.meta && input.key.toLowerCase() === 'h' && input.type === 'keyDown') {
            event.preventDefault();
            mainWindow.makeInvisible();
        }
    });

    // Before the window is destroyed, sign the user out via Clerk in the renderer.
    // This covers: red-X close, Cmd+Q quit, and any other close trigger.
    // We preventDefault(), send force-logout, then wait for the renderer to
    // call signOutComplete() — at that point we destroy() the window (which
    // skips this listener and lets the normal quit/close flow continue).
    mainWindow.on('close', (event) => {
        if (mainWindow._logoutDone) return; // already signed out — let it proceed
        event.preventDefault();
        mainWindow.webContents.send('force-logout');
        // Safety net: if renderer doesn't respond in 3 s (crash/not loaded), destroy anyway
        mainWindow._logoutTimer = setTimeout(() => {
            if (mainWindow) { mainWindow._logoutDone = true; mainWindow.destroy(); }
        }, 3000);
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Track invisible state on the window object so the globalShortcut can access it
    mainWindow._invisible = false;
    mainWindow._savedBounds = null;

    mainWindow.makeInvisible = () => {
        if (mainWindow._invisible) return;
        mainWindow._invisible = true;
        // Save current position, then move far off-screen.
        // macOS ScreenCaptureKit (OBS Window Capture) still captures the window
        // content by its window ID even when it's off-screen — the feed stays live.
        mainWindow._savedBounds = mainWindow.getBounds();
        mainWindow.setPosition(-9999, -9999);
        if (process.platform === 'darwin') app.dock.hide();
        console.log('[Electron] Window moved off-screen — OBS Window Capture still active');
    };

    mainWindow.makeVisible = () => {
        if (!mainWindow._invisible) return;
        mainWindow._invisible = false;
        if (process.platform === 'darwin') app.dock.show();
        // Restore to original position
        if (mainWindow._savedBounds) {
            const b = mainWindow._savedBounds;
            mainWindow.setBounds(b);
        }
        mainWindow.focus();
        console.log('[Electron] Window restored');
    };
}

app.whenReady().then(async () => {
    // Start the static server once here so it survives window close/reopen cycles.
    // createWindow() is called multiple times on macOS (Dock click), but the
    // server must only bind the port once — otherwise EADDRINUSE → file:// fallback → Clerk 401.
    if (app.isPackaged) {
        const distPath = path.join(__dirname, '../dist');
        await startStaticServer(distPath).catch(err =>
            console.error('[Electron] Static server failed to start:', err)
        );
    }

    createWindow();

    // Global hotkey: Cmd+Shift+H — works even when the app is invisible & unfocused
    globalShortcut.register('CommandOrControl+Shift+H', () => {
        if (!mainWindow) return;
        mainWindow._invisible ? mainWindow.makeVisible() : mainWindow.makeInvisible();
    });

    console.log('[Electron] Global hotkey registered: Cmd+Shift+H to toggle visibility');
});

app.on('will-quit', () => {
    globalShortcut.unregisterAll();
    if (staticServer) staticServer.close();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    } else {
        // Clicking the Dock icon brings the window back
        mainWindow.makeVisible?.();
    }
});

// Renderer calls this after Clerk signOut() completes — we then destroy the window
ipcMain.on('logout-complete', () => {
    if (!mainWindow) return;
    clearTimeout(mainWindow._logoutTimer);
    mainWindow._logoutDone = true;
    mainWindow.destroy();
});

ipcMain.handle('install-virtual-mic', async () => {
    return new Promise((resolve) => {
        if (process.platform !== 'darwin') {
            resolve({ success: false, error: 'Only supported on macOS currently.' });
            return;
        }

        console.log('[Electron] Installing BlackHole via Homebrew...');

        // Use Homebrew to install BlackHole - handles macOS compatibility automatically
        const brewPath = '/opt/homebrew/bin/brew';
        exec(brewPath + ' install blackhole-2ch', { timeout: 120000 }, (error, stdout, stderr) => {
            if (error) {
                console.error('[Electron] Homebrew install failed:', error.message);
                console.error('[Electron] stderr:', stderr);
                // Check if already installed
                if (stderr && stderr.includes('already installed')) {
                    console.log('[Electron] BlackHole is already installed!');
                    resolve({ success: true, alreadyInstalled: true });
                    return;
                }
                resolve({ success: false, error: error.message });
                return;
            }
            console.log('[Electron] Homebrew install stdout:', stdout);
            resolve({ success: true });
        });
    });
});
