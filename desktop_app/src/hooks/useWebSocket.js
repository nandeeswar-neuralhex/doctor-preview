import { useState, useEffect, useRef, useCallback } from 'react';

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
            wsRef.current = ws;

            ws.onopen = () => {
                console.log('WebSocket Connected');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                // If it's a JSON message
                if (event.data.startsWith('{') && event.data.endsWith('}')) {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.error) {
                            setError(data.error);
                        }
                        // Handle image response with timestamp
                        if (data.image) {
                            const latency = data.ts ? Date.now() - data.ts : 0;
                            if (onFrame) onFrame(data.image, latency);
                        }
                    } catch (e) {
                        // ignore
                    }
                } else {
                    // Legacy: raw base64 frame
                    if (onFrame) {
                        onFrame(event.data, 0);
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

    const sendFrame = useCallback((base64Frame) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            // Send JSON with timestamp for latency tracking
            const payload = JSON.stringify({
                image: base64Frame,
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
    }, []); // Removed connect/disconnect from dependency array to avoid loop

    return { connect, disconnect, sendFrame, isConnected, error };
};

export default useWebSocket;
