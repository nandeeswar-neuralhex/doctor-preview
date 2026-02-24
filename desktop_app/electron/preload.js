const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    installVirtualMic: () => ipcRenderer.invoke('install-virtual-mic')
});
