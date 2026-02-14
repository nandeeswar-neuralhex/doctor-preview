// Polyfill crypto.getRandomValues for Node.js < 20
// Must run BEFORE Vite starts
const { webcrypto } = require('crypto');

if (!globalThis.crypto) {
    globalThis.crypto = webcrypto;
} else if (!globalThis.crypto.getRandomValues) {
    globalThis.crypto.getRandomValues = webcrypto.getRandomValues.bind(webcrypto);
}
