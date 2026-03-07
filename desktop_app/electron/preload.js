const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    installVirtualMic: () => ipcRenderer.invoke('install-virtual-mic'),
    // Cmd+R / Cmd+Shift+R intercepted by main — triggers logout instead of reload
    onTriggerLogout: (callback) => ipcRenderer.on('trigger-logout', callback),
    removeTriggerLogout: (callback) => ipcRenderer.removeListener('trigger-logout', callback),
    // Window/app closing — main asks renderer to sign out before destroying window
    onForceLogout: (callback) => ipcRenderer.on('force-logout', callback),
    removeForceLogout: (callback) => ipcRenderer.removeListener('force-logout', callback),
    // Renderer calls this once Clerk signOut() has finished
    signOutComplete: () => ipcRenderer.send('logout-complete'),
});
