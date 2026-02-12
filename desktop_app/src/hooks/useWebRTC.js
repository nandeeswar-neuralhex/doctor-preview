import { useCallback, useRef, useState } from 'react';

function useWebRTC(serverUrl, sessionId, onRemoteStream) {
    const pcRef = useRef(null);
    const [error, setError] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [connectionState, setConnectionState] = useState('new');

    const connect = useCallback(async (localStream) => {
        try {
            if (!serverUrl) {
                setError('Server URL not set');
                return;
            }

            const pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            pc.ontrack = (event) => {
                const [remoteStream] = event.streams;
                if (onRemoteStream) {
                    onRemoteStream(remoteStream);
                }
            };

            pc.onconnectionstatechange = () => {
                setConnectionState(pc.connectionState);
                if (pc.connectionState === 'connected') {
                    setIsConnected(true);
                }
                if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
                    setIsConnected(false);
                }
            };

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            await new Promise((resolve) => {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    // Timeout after 5 seconds to avoid hanging on poor network
                    const timer = setTimeout(() => {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }, 5000);
                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
                            clearTimeout(timer);
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    };
                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });

            const response = await fetch(`${serverUrl}/webrtc/offer?session_id=${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
            });

            if (!response.ok) {
                const body = await response.json().catch(() => null);
                const message = body?.error || 'WebRTC offer failed';
                throw new Error(`${message} (HTTP ${response.status})`);
            }

            const answer = await response.json();
            await pc.setRemoteDescription(answer);

            pcRef.current = pc;
            setError(null);
        } catch (e) {
            setError(e.message || 'WebRTC error');
            setIsConnected(false);
            setConnectionState('failed');
        }
    }, [serverUrl, sessionId, onRemoteStream]);

    const disconnect = useCallback(() => {
        if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
        }
        // Notify server to clean up session resources
        if (serverUrl && sessionId) {
            fetch(`${serverUrl}/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
        }
        setIsConnected(false);
        setConnectionState('closed');
    }, [serverUrl, sessionId]);

    return { connect, disconnect, isConnected, error, connectionState };
}

export default useWebRTC;