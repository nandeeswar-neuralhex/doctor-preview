import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import crypto from 'crypto';

// Polyfill for Node.js < 20
if (!globalThis.crypto) {
    globalThis.crypto = crypto.webcrypto;
}

export default defineConfig({
    plugins: [react()],
    base: './',
    server: {
        port: 3000
    },
    build: {
        outDir: 'dist'
    }
});
