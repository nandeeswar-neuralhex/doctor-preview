const { app, BrowserWindow } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false,
            webSecurity: false          // Allow cross-origin requests (Vite HMR + RunPod)
        },
        title: 'Doctor Preview - Surgery Preview System',
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
    mainWindow.webContents.on('before-input-event', (event, input) => {
        if (input.key === 'F12' ||
            (input.meta && input.shift && input.key.toLowerCase() === 'i')) {
            mainWindow.webContents.toggleDevTools();
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
