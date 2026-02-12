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
                    const checkState = () => {
                        if (pc.iceGatheringState === 'complete') {
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
        setIsConnected(false);
        setConnectionState('closed');
    }, []);

    return { connect, disconnect, isConnected, error, connectionState };
}

export default useWebRTC;