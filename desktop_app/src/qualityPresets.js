/**
 * YouTube-style quality presets for doctor to manually adjust.
 * Each preset controls: resolution, bitrate (WebRTC), JPEG quality & FPS (WebSocket).
 */
const QUALITY_PRESETS = {
    '1080p': {
        label: '1080p HD',
        width: 1920,
        height: 1080,
        maxBitrate: 4_000_000,   // 4 Mbps — crisp HD over WebRTC
        jpegQuality: 0.92,       // WebSocket JPEG quality
        fps: 30,
        icon: 'HD',
    },
    '720p': {
        label: '720p',
        width: 1280,
        height: 720,
        maxBitrate: 2_500_000,   // 2.5 Mbps
        jpegQuality: 0.85,
        fps: 30,
        icon: 'HD',
    },
    '480p': {
        label: '480p',
        width: 854,
        height: 480,
        maxBitrate: 1_000_000,   // 1 Mbps
        jpegQuality: 0.78,
        fps: 24,
        icon: '',
    },
    '360p': {
        label: '360p',
        width: 640,
        height: 360,
        maxBitrate: 500_000,     // 500 Kbps — low bandwidth
        jpegQuality: 0.65,
        fps: 20,
        icon: '',
    },
    'auto': {
        label: 'Auto',
        width: 1280,
        height: 720,
        maxBitrate: 2_500_000,
        jpegQuality: 0.80,
        fps: 24,
        icon: '⚙',
    },
};

export const DEFAULT_QUALITY = '720p';
export default QUALITY_PRESETS;
