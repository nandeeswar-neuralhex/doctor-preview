// Polyfill crypto.getRandomValues for Node.js < 20
// Must run BEFORE Vite starts
const nodeCrypto = require('crypto');
const { webcrypto } = nodeCrypto;

// Ensure globalThis.crypto is the Web Crypto API
if (!globalThis.crypto || !globalThis.crypto.getRandomValues) {
    globalThis.crypto = webcrypto;
}

// Patch the CJS `require('crypto')` module so that Vite's bundled code
// (which does `import crypto from 'crypto'`) also gets getRandomValues
if (!nodeCrypto.getRandomValues) {
    nodeCrypto.getRandomValues = webcrypto.getRandomValues.bind(webcrypto);
}
