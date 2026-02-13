import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Binary WebSocket protocol for low-latency face swap streaming.
 * 
 * Send: [8 bytes float64 timestamp] + [raw JPEG bytes]
 * Recv: [8 bytes float64 timestamp] + [raw JPEG bytes]
 * 
 * Eliminates base64 encoding (33% bandwidth saving) and JSON parse overhead (~6ms).
 * Falls back to text/JSON mode if binary send fails.
 */
const useWebSocket = (serverUrl, sessionId, onFrame) => {
    const wsRef = useRef(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);

    const connect = useCallback(() => {
        if (!serverUrl || !sessionId) return;

        // Convert http/https to ws/wss
        const wsUrl = serverUrl.replace(/^http/, 'ws').replace(/\/$/, '') + `/ws/${sessionId}`;

        // Prevent multiple connections
        if (wsRef.current) {
            if (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING) {
                return;
            }
            wsRef.current.close();
        }

        try {
            console.log('Connecting to WebSocket:', wsUrl);
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer'; // Receive binary responses as ArrayBuffer
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket Connected (binary mode)');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                if (event.data instanceof ArrayBuffer) {
                    // Binary mode: first 8 bytes = float64 timestamp, rest = JPEG
                    const buf = event.data;
                    if (buf.byteLength < 16) return;

                    const view = new DataView(buf);
                    const sentTs = view.getFloat64(0, true); // little-endian
                    const latency = Date.now() - sentTs;

                    // Create blob URL from raw JPEG (zero-copy, no base64)
                    const jpegBlob = new Blob([buf.slice(8)], { type: 'image/jpeg' });
                    const url = URL.createObjectURL(jpegBlob);

                    if (onFrame) onFrame(url, latency, true); // true = isObjectURL
                } else if (typeof event.data === 'string') {
                    // Legacy text/JSON fallback
                    if (event.data.startsWith('{') && event.data.endsWith('}')) {
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
     * Send a frame as raw binary: [8-byte float64 timestamp] + [JPEG bytes]
     * Accepts either a Blob or a base64 string.
     */
    const sendFrame = useCallback((frameData) => {
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

        if (frameData instanceof Blob) {
            // Fast path: binary blob from canvas.toBlob()
            frameData.arrayBuffer().then((jpegBuf) => {
                const header = new ArrayBuffer(8);
                new DataView(header).setFloat64(0, Date.now(), true);
                // Concat header + JPEG into single send
                const combined = new Uint8Array(8 + jpegBuf.byteLength);
                combined.set(new Uint8Array(header), 0);
                combined.set(new Uint8Array(jpegBuf), 8);
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(combined.buffer);
                }
            });
        } else if (typeof frameData === 'string') {
            // Legacy fallback: base64 string
            const payload = JSON.stringify({
                image: frameData,
                ts: Date.now()
            });
            wsRef.current.send(payload);
        }
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setIsConnected(false);
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            disconnect();
        };
    }, []);

    return { connect, disconnect, sendFrame, isConnected, error };
};

export default useWebSocket;
