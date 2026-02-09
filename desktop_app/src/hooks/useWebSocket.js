import { useState, useCallback, useRef } from 'react';

function useWebSocket(serverUrl, sessionId, onFrameReceived) {
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const sendTimeRef = useRef(null);

    const connect = useCallback(async () => {
        if (!serverUrl) {
            setError('Server URL not set');
            return;
        }

        try {
            // Convert HTTP URL to WebSocket URL
            const wsUrl = serverUrl
                .replace('https://', 'wss://')
                .replace('http://', 'ws://');

            const ws = new WebSocket(`${wsUrl}/ws/${sessionId}`);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setError(null);
            };

            ws.onmessage = (event) => {
                const receiveTime = Date.now();
                const latency = sendTimeRef.current ? receiveTime - sendTimeRef.current : 0;

                // Call callback with processed frame and latency
                if (onFrameReceived) {
                    onFrameReceived(event.data, latency);
                }
            };

            ws.onerror = (err) => {
                console.error('WebSocket error:', err);
                setError('WebSocket connection error');
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setIsConnected(false);
            };

            wsRef.current = ws;
        } catch (err) {
            setError(`Failed to connect: ${err.message}`);
            console.error('WebSocket connection error:', err);
        }
    }, [serverUrl, sessionId, onFrameReceived]);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
            setIsConnected(false);
        }
    }, []);

    const sendFrame = useCallback((base64Frame) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            sendTimeRef.current = Date.now();
            wsRef.current.send(base64Frame);
        }
    }, []);

    return { isConnected, error, sendFrame, connect, disconnect };
}

export default useWebSocket;
