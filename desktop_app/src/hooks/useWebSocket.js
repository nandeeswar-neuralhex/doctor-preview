import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Adaptive binary WebSocket for real-time face swap streaming.
 *
 * Protocol (binary):
 *   Send: [4B frameId] + [8B timestamp] + [4B audioLen] + [4B sampleRate] + [audio PCM] + [JPEG]
 *   Recv: [4B frameId] + [8B timestamp] + [raw JPEG bytes]
 *
 * Key improvements (Google-Meet-grade):
 * 1. ADAPTIVE FPS — scales send rate based on bufferedAmount backpressure
 * 2. ADAPTIVE QUALITY — lowers JPEG quality when bandwidth is constrained
 * 3. PIPELINED sending — fires frames without waiting for responses
 * 4. Exposes networkQuality for UI indicators
 */
const useWebSocket = (serverUrl, sessionId, onFrame) => {
    const wsRef = useRef(null);
    const frameIdRef = useRef(0);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    // Network quality: 'excellent' | 'good' | 'fair' | 'poor'
    const [networkQuality, setNetworkQuality] = useState('good');

    // ── Adaptive send-rate state (refs for use in tight loops) ──
    const adaptiveRef = useRef({
        currentFps: 20,
        currentQuality: 0.75,
        targetFps: 20,
        targetQuality: 0.75,
        maxBuffered: 1_000_000,
        avgBuffered: 0,
        droppedFrames: 0,
        sentFrames: 0,
    });

    /**
     * Configure the adaptive sender from a hardware profile.
     * Call this after getHardwareProfile() resolves.
     */
    const configureFromProfile = useCallback((profile) => {
        if (!profile) return;
        const a = adaptiveRef.current;
        a.currentFps = profile.encode.sendFps;
        a.targetFps = profile.encode.sendFps;
        a.currentQuality = profile.encode.jpegQuality;
        a.targetQuality = profile.encode.jpegQuality;
        a.maxBuffered = profile.encode.maxBufferedBytes;
        console.log(`[WS-Adaptive] Configured: ${a.currentFps}fps, q=${a.currentQuality}, maxBuf=${a.maxBuffered}`);
    }, []);

    const connect = useCallback(() => {
        if (!serverUrl || !sessionId) return;

        const wsUrl = serverUrl.replace(/^http/, 'ws').replace(/\/$/, '') + `/ws/${sessionId}`;

        if (wsRef.current) {
            const state = wsRef.current.readyState;
            if (state === WebSocket.OPEN || state === WebSocket.CONNECTING) {
                console.log('WebSocket already connected/connecting, skipping');
                return;
            }
            try { wsRef.current.close(); } catch (_) { }
            wsRef.current = null;
        }

        try {
            console.log('Connecting to WebSocket:', wsUrl);
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onopen = () => {
                if (wsRef.current !== ws) {
                    ws.close();
                    return;
                }
                console.log('WebSocket Connected (adaptive binary pipeline)');
                setIsConnected(true);
                setError(null);
                frameIdRef.current = 0;
                const a = adaptiveRef.current;
                a.droppedFrames = 0;
                a.sentFrames = 0;
                a.avgBuffered = 0;
            };

            ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    const buf = event.data;
                    if (buf.byteLength < 12) return;

                    const view = new DataView(buf);
                    const sentTs = view.getFloat64(4, true);
                    const latency = Date.now() - sentTs;

                    const jpegBlob = new Blob([buf.slice(12)], { type: 'image/jpeg' });
                    if (onFrame) onFrame(jpegBlob, latency, true);
                } else if (typeof event.data === 'string') {
                    if (event.data.startsWith('{')) {
                        try {
                            const data = JSON.parse(event.data);
                            if (data.error) setError(data.error);
                            if (data.image) {
                                const latency = data.ts ? Date.now() - data.ts : 0;
                                if (onFrame) onFrame(`data:image/jpeg;base64,${data.image}`, latency, false);
                            }
                        } catch (e) { /* ignore */ }
                    } else {
                        if (onFrame) onFrame(`data:image/jpeg;base64,${event.data}`, 0, false);
                    }
                }
            };

            ws.onclose = () => {
                console.log('WebSocket Disconnected');
                setIsConnected(false);
            };

            ws.onerror = (e) => {
                console.error('WebSocket Error:', e);
                setError('WebSocket connection failed');
                setIsConnected(false);
            };

        } catch (e) {
            setError(e.message);
        }
    }, [serverUrl, sessionId, onFrame]);

    /**
     * Adaptive backpressure engine — called before each frame send.
     * Monitors bufferedAmount and adjusts FPS + quality in real time.
     * This is the "Google Meet" adaptation — works on any network/hardware.
     */
    const adaptSendRate = useCallback(() => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const a = adaptiveRef.current;
        const buffered = ws.bufferedAmount;

        // Exponential moving average of buffered bytes
        a.avgBuffered = a.avgBuffered * 0.7 + buffered * 0.3;

        const utilization = a.avgBuffered / a.maxBuffered;

        let quality;
        if (utilization > 0.8) {
            a.currentFps = Math.max(6, a.targetFps * 0.4);
            quality = 'poor';
        } else if (utilization > 0.5) {
            a.currentFps = Math.max(8, a.targetFps * 0.65);
            quality = 'fair';
        } else if (utilization > 0.2) {
            a.currentFps = Math.max(12, a.targetFps * 0.85);
            quality = 'good';
        } else {
            a.currentFps = a.targetFps;
            quality = 'excellent';
        }

        // Scale JPEG quality with congestion
        if (utilization > 0.6) {
            a.currentQuality = Math.max(0.40, a.targetQuality - 0.20);
        } else if (utilization > 0.3) {
            a.currentQuality = Math.max(0.50, a.targetQuality - 0.10);
        } else {
            a.currentQuality = a.targetQuality;
        }

        setNetworkQuality(quality);
    }, []);

    /**
     * Send frame as binary with audio for lip sync.
     * PIPELINED: fires immediately, does NOT wait for response.
     * Returns false if frame was dropped (backpressure).
     */
    const sendFrame = useCallback((frameData, audioData, sampleRate) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return false;

        const a = adaptiveRef.current;

        // Backpressure gate — drop frame if buffer is overwhelmed
        if (wsRef.current.bufferedAmount > a.maxBuffered) {
            a.droppedFrames++;
            return false;
        }

        a.sentFrames++;
        const fid = frameIdRef.current++;

        if (frameData instanceof Blob) {
            frameData.arrayBuffer().then((jpegBuf) => {
                if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

                const audioPcm = audioData instanceof Int16Array && audioData.length > 0
                    ? audioData : new Int16Array(0);
                const audioBytes = new Uint8Array(
                    audioPcm.buffer, audioPcm.byteOffset, audioPcm.byteLength
                );

                const headerSize = 20;
                const totalSize = headerSize + audioBytes.length + jpegBuf.byteLength;
                const combined = new Uint8Array(totalSize);
                const view = new DataView(combined.buffer);
                view.setUint32(0, fid, true);
                view.setFloat64(4, Date.now(), true);
                view.setUint32(12, audioBytes.length, true);
                view.setUint32(16, sampleRate || 16000, true);
                combined.set(audioBytes, headerSize);
                combined.set(new Uint8Array(jpegBuf), headerSize + audioBytes.length);

                wsRef.current.send(combined.buffer);
            });
        } else if (typeof frameData === 'string') {
            wsRef.current.send(JSON.stringify({ image: frameData, ts: Date.now() }));
        }
        return true;
    }, []);

    /**
     * Get current adaptive parameters (for use in the send loop).
     */
    const getAdaptiveParams = useCallback(() => {
        return adaptiveRef.current;
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    useEffect(() => {
        return () => { disconnect(); };
    }, []);

    return {
        connect,
        disconnect,
        sendFrame,
        isConnected,
        error,
        networkQuality,
        configureFromProfile,
        adaptSendRate,
        getAdaptiveParams,
    };
};

export default useWebSocket;
