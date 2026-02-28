const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { exec } = require('child_process');

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
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
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
        // Dev branch: allow normal Cmd+R refresh (logout-on-refresh disabled)
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});

ipcMain.handle('install-virtual-mic', async () => {
    return new Promise((resolve) => {
        const platform = process.platform;

        if (platform === 'darwin') {
            // macOS: Install BlackHole via Homebrew
            console.log('[Electron] Installing BlackHole via Homebrew...');
            const brewPaths = ['/opt/homebrew/bin/brew', '/usr/local/bin/brew'];
            const brewPath = brewPaths.find(p => {
                try { require('fs').accessSync(p); return true; } catch { return false; }
            }) || 'brew';

            exec(brewPath + ' install blackhole-2ch', { timeout: 120000 }, (error, stdout, stderr) => {
                if (error) {
                    console.error('[Electron] Homebrew install failed:', error.message);
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

        } else if (platform === 'win32') {
            // Windows: Check if VB-Audio Virtual Cable is installed, guide user if not
            console.log('[Electron] Checking for VB-Audio Virtual Cable on Windows...');
            exec('reg query "HKLM\\SOFTWARE\\VB-Audio" /s', { timeout: 10000 }, (error) => {
                if (error) {
                    // VB-Audio not found in registry — guide user to install
                    console.log('[Electron] VB-Audio not found. Opening download page...');
                    const { shell } = require('electron');
                    shell.openExternal('https://vb-audio.com/Cable/');
                    resolve({
                        success: false,
                        error: 'VB-Audio Virtual Cable not detected. Download page opened — install it and restart the app.',
                        downloadUrl: 'https://vb-audio.com/Cable/'
                    });
                } else {
                    console.log('[Electron] VB-Audio Virtual Cable is installed!');
                    resolve({ success: true, alreadyInstalled: true });
                }
            });

        } else {
            // Linux: Create a PulseAudio null sink
            console.log('[Electron] Setting up PulseAudio null sink on Linux...');
            exec('pactl load-module module-null-sink sink_name=DoctorPreview sink_properties=device.description=DoctorPreview', { timeout: 10000 }, (error, stdout) => {
                if (error) {
                    console.error('[Electron] PulseAudio setup failed:', error.message);
                    resolve({ success: false, error: `PulseAudio error: ${error.message}` });
                    return;
                }
                console.log('[Electron] PulseAudio null sink created:', stdout);
                resolve({ success: true });
            });
        }
    });
});
