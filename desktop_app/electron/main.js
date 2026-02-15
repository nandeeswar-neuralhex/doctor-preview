const { app, BrowserWindow, globalShortcut } = require('electron');
const path = require('path');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            sandbox: false
        },
        title: 'Doctor Preview - Surgery Preview System',
        backgroundColor: '#1a1a1a'
    });

    // Load from Vite dev server in development, or built files in production
    const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

    if (isDev) {
        mainWindow.loadURL('http://localhost:3000');
        // Don't auto-open DevTools — it crashes on Electron 28 + macOS
        // Use Cmd+Shift+I or F12 to open manually when needed
    } else {
        mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
    }

    // Open DevTools on demand via keyboard shortcut
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
