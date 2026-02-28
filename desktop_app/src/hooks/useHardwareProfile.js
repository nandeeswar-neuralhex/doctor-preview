/**
 * Hardware-adaptive profile detection — like Google Meet's quality selection.
 *
 * Detects device capabilities and returns optimal capture/encode settings.
 * Runs a fast benchmark on first call to measure canvas.toBlob() throughput,
 * then caches the result for the session.
 *
 * Tiers:
 *   HIGH   — Apple Silicon, modern desktop GPUs (8+ cores, 8+ GB RAM)
 *   MEDIUM — Mid-range laptops, Intel i5/Ryzen 5 (4+ cores, 4+ GB RAM)
 *   LOW    — Budget laptops, old desktops, Celeron/Pentium (< 4 cores)
 */

let _cachedProfile = null;

/**
 * Benchmark canvas.toBlob('image/jpeg') to measure actual encode speed.
 * Returns average ms per encode over 5 frames at 640x480.
 */
async function benchmarkJpegEncode() {
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');

    // Draw a non-trivial pattern (not blank — blank compresses instantly)
    for (let i = 0; i < 20; i++) {
        ctx.fillStyle = `hsl(${i * 18}, 70%, 50%)`;
        ctx.fillRect(Math.random() * 600, Math.random() * 440, 80, 60);
    }

    const times = [];
    for (let i = 0; i < 5; i++) {
        const t0 = performance.now();
        await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.7);
        });
        times.push(performance.now() - t0);
    }

    // Drop first (cold) and take average of rest
    times.shift();
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    return avg;
}

/**
 * Detect if WebCodecs VideoEncoder is available with JPEG support.
 * Electron 28+ (Chromium 120) supports this.
 */
function hasWebCodecsJpeg() {
    // We actually need the canvas path since server expects JPEG,
    // but we can check if OffscreenCanvas + convertToBlob is available
    // (faster than canvas.toBlob in workers)
    return typeof OffscreenCanvas !== 'undefined';
}

/**
 * Detect platform-specific capabilities.
 */
function detectPlatform() {
    const ua = navigator.userAgent || '';
    const platform = navigator.platform || '';

    const isAppleSilicon = /Mac/.test(platform) && (navigator.hardwareConcurrency || 0) >= 8;
    const isMac = /Mac/.test(platform);
    const isWindows = /Win/.test(platform);
    const isLinux = /Linux/.test(platform) && !/Android/.test(ua);

    return { isAppleSilicon, isMac, isWindows, isLinux };
}

/**
 * Get the hardware profile with optimal settings.
 * First call runs a benchmark (~100ms), subsequent calls return cached result.
 */
export async function getHardwareProfile() {
    if (_cachedProfile) return _cachedProfile;

    const cores = navigator.hardwareConcurrency || 2;
    const memory = navigator.deviceMemory || 2; // GB (Chrome-only, defaults to 2)
    const { isAppleSilicon, isMac, isWindows, isLinux } = detectPlatform();

    // Benchmark actual JPEG encode speed
    let encodeMsAvg;
    try {
        encodeMsAvg = await benchmarkJpegEncode();
    } catch (e) {
        encodeMsAvg = 50; // Assume slow if benchmark fails
    }

    const hasOffscreenCanvas = hasWebCodecsJpeg();

    // Determine tier based on hardware signals + actual benchmark
    let tier, captureWidth, captureHeight, sendFps, jpegQuality, maxBufferedBytes;

    if (isAppleSilicon || (encodeMsAvg < 8 && cores >= 8 && memory >= 8)) {
        // HIGH: Apple Silicon or fast desktop with proven fast encode
        tier = 'high';
        captureWidth = 1280;
        captureHeight = 720;
        sendFps = 24;
        jpegQuality = 0.80;
        maxBufferedBytes = 1_500_000; // 1.5MB — allow more in-flight
    } else if (encodeMsAvg < 20 && cores >= 4 && memory >= 4) {
        // MEDIUM: decent laptop, encode is manageable
        tier = 'medium';
        captureWidth = 854;
        captureHeight = 480;
        sendFps = 18;
        jpegQuality = 0.70;
        maxBufferedBytes = 800_000;
    } else {
        // LOW: slow encode, few cores, limited RAM
        tier = 'low';
        captureWidth = 640;
        captureHeight = 360;
        sendFps = 12;
        jpegQuality = 0.55;
        maxBufferedBytes = 400_000;
    }

    _cachedProfile = {
        tier,
        cores,
        memory,
        isAppleSilicon,
        isMac,
        isWindows,
        isLinux,
        encodeMsAvg: Math.round(encodeMsAvg * 10) / 10,
        hasOffscreenCanvas,
        capture: {
            width: captureWidth,
            height: captureHeight,
        },
        encode: {
            sendFps,
            jpegQuality,
            maxBufferedBytes,
        },
    };

    console.log(`[HardwareProfile] Tier: ${tier.toUpperCase()}`);
    console.log(`  Cores: ${cores}, Memory: ${memory}GB, Apple Silicon: ${isAppleSilicon}`);
    console.log(`  JPEG encode benchmark: ${encodeMsAvg.toFixed(1)}ms @ 640x480`);
    console.log(`  → Capture: ${captureWidth}x${captureHeight}, Send: ${sendFps}fps, Quality: ${jpegQuality}`);
    console.log(`  OffscreenCanvas: ${hasOffscreenCanvas}`);

    return _cachedProfile;
}

/**
 * Get cached profile synchronously (returns null if not yet detected).
 */
export function getCachedProfile() {
    return _cachedProfile;
}

export default getHardwareProfile;
