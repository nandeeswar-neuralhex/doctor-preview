import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * High-performance binary WebSocket for real-time face swap streaming.
 *
 * Protocol (binary):
 *   Send: [4 bytes uint32 frame_id] + [raw JPEG bytes]
 *   Recv: [4 bytes uint32 frame_id] + [raw JPEG bytes]
 *
 * Key design: PIPELINED sending — we do NOT wait for a response before sending
 * the next frame. This hides the network latency completely. The server processes
 * frames in order and we display responses as they arrive (latest wins).
 *
 * With ~300ms round-trip and pipelined sending at 20fps, there are always
 * ~6 frames "in flight". The user sees smooth 20fps output despite high RTT.
 */
const useWebSocket = (serverUrl, sessionId, onFrame) => {
    const wsRef = useRef(null);
    const frameIdRef = useRef(0);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);

    const connect = useCallback(() => {
        if (!serverUrl || !sessionId) return;

        const wsUrl = serverUrl.replace(/^http/, 'ws').replace(/\/$/, '') + `/ws/${sessionId}`;

        if (wsRef.current) {
            if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
                return;
            }
            wsRef.current.close();
        }

        try {
            console.log('Connecting to WebSocket:', wsUrl);
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket Connected (binary pipeline mode)');
                setIsConnected(true);
                setError(null);
                frameIdRef.current = 0;
            };

            ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    const buf = event.data;
                    if (buf.byteLength < 12) return;

                    // Header: 4 bytes frameId + 8 bytes float64 timestamp
                    const view = new DataView(buf);
                    const sentTs = view.getFloat64(4, true);
                    const latency = Date.now() - sentTs;

                    // Pass raw Blob directly — CameraView uses createImageBitmap(blob)
                    // No blob URL creation/revocation overhead
                    const jpegBlob = new Blob([buf.slice(12)], { type: 'image/jpeg' });
                    if (onFrame) onFrame(jpegBlob, latency, true);
                } else if (typeof event.data === 'string') {
                    // Legacy text fallback
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
     * Send frame as binary with audio for lip sync:
     * [4B frameId] + [8B timestamp] + [4B audioLen] + [4B sampleRate] + [audio PCM] + [JPEG]
     * PIPELINED: does NOT wait for previous response — fires immediately.
     */
    const sendFrame = useCallback((frameData, audioData, sampleRate) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        // Check bufferedAmount — if too much is queued, skip this frame
        // This prevents memory buildup if network is slower than capture rate
        if (wsRef.current.bufferedAmount > 200_000) {
            return; // Skip frame, ~200KB already queued
        }

        const fid = frameIdRef.current++;

        if (frameData instanceof Blob) {
            frameData.arrayBuffer().then((jpegBuf) => {
                if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

                // Audio PCM bytes (Int16Array → raw bytes)
                const audioPcm = audioData instanceof Int16Array && audioData.length > 0
                    ? audioData : new Int16Array(0);
                const audioBytes = new Uint8Array(
                    audioPcm.buffer, audioPcm.byteOffset, audioPcm.byteLength
                );

                // Header: 4B frameId + 8B timestamp + 4B audioLen + 4B sampleRate = 20 bytes
                const headerSize = 20;
                const totalSize = headerSize + audioBytes.length + jpegBuf.byteLength;
                const combined = new Uint8Array(totalSize);
                const view = new DataView(combined.buffer);
                view.setUint32(0, fid, true);                // frameId
                view.setFloat64(4, Date.now(), true);         // timestamp
                view.setUint32(12, audioBytes.length, true);  // audio byte length
                view.setUint32(16, sampleRate || 16000, true); // sample rate
                combined.set(audioBytes, headerSize);
                combined.set(new Uint8Array(jpegBuf), headerSize + audioBytes.length);

                wsRef.current.send(combined.buffer);
            });
        } else if (typeof frameData === 'string') {
            // Legacy text fallback
            wsRef.current.send(JSON.stringify({ image: frameData, ts: Date.now() }));
        }
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

    return { connect, disconnect, sendFrame, isConnected, error };
};

export default useWebSocket;
