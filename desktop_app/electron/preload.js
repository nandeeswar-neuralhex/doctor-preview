const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    installVirtualMic: () => ipcRenderer.invoke('install-virtual-mic'),
    onTriggerLogout: (callback) => ipcRenderer.on('trigger-logout', callback),
    removeTriggerLogout: (callback) => ipcRenderer.removeListener('trigger-logout', callback)
});
