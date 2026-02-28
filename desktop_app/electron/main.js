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
        // Cmd+R or Cmd+Shift+R → trigger logout instead of page refresh
        if (input.meta && (input.key.toLowerCase() === 'r') && input.type === 'keyDown') {
            event.preventDefault();
            mainWindow.webContents.send('trigger-logout');
        }
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
