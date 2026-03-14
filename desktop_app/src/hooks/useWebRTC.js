import { useCallback, useRef, useState } from 'react';
import QUALITY_PRESETS from '../qualityPresets';

function useWebRTC(serverUrl, sessionId, onRemoteStream) {
    const pcRef = useRef(null);
    const [error, setError] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [connectionState, setConnectionState] = useState('new');

    /**
     * Apply bitrate / resolution constraints to all video senders.
     * Called after connect and whenever the quality preset changes.
     */
    const applyQuality = useCallback(async (presetKey) => {
        const pc = pcRef.current;
        if (!pc) return;
        const preset = QUALITY_PRESETS[presetKey];
        if (!preset) return;

        const senders = pc.getSenders().filter(s => s.track?.kind === 'video');
        for (const sender of senders) {
            try {
                const params = sender.getParameters();
                if (!params.encodings || params.encodings.length === 0) {
                    params.encodings = [{}];
                }
                params.encodings[0].maxBitrate = preset.maxBitrate;
                // scaleResolutionDownBy: scale the outgoing video down from its
                // native resolution to match the preset height.
                // e.g. 720p camera sending at 480p → scale = 720/480 = 1.5
                const track = sender.track;
                if (track) {
                    const settings = track.getSettings();
                    const nativeH = settings.height || 720;
                    const scale = Math.max(1, nativeH / preset.height);
                    params.encodings[0].scaleResolutionDownBy = scale;
                }
                // Max framerate
                params.encodings[0].maxFramerate = preset.fps;
                await sender.setParameters(params);
            } catch (_) {
                // Some browsers don't support all parameters — safe to ignore
            }
        }
    }, []);

    const connect = useCallback(async (localStream, qualityPreset) => {
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

            // Inject b=AS bandwidth hint into the SDP for the server side
            const preset = QUALITY_PRESETS[qualityPreset || '720p'];
            let sdp = offer.sdp;
            if (preset) {
                const bwKbps = Math.round(preset.maxBitrate / 1000);
                // Add bandwidth line after each m=video line
                sdp = sdp.replace(
                    /(m=video.*\r\n)/g,
                    `$1b=AS:${bwKbps}\r\n`
                );
            }
            await pc.setLocalDescription({ type: offer.type, sdp });

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

            // Apply sender-side bitrate/resolution constraints
            if (qualityPreset) {
                // Small delay to let encoders initialize
                setTimeout(() => applyQuality(qualityPreset), 500);
            }
        } catch (e) {
            setError(e.message || 'WebRTC error');
            setIsConnected(false);
            setConnectionState('failed');
            throw e; // Re-throw so callers can catch and fall back
        }
    }, [serverUrl, sessionId, onRemoteStream, applyQuality]);

    const disconnect = useCallback(() => {
        if (pcRef.current) {
            pcRef.current.close();
            pcRef.current = null;
        }
        // Notify server to clean up session resources
        if (serverUrl && sessionId) {
            fetch(`${serverUrl}/session/${sessionId}`, { method: 'DELETE' }).catch(() => { });
        }
        setIsConnected(false);
        setConnectionState('closed');
    }, [serverUrl, sessionId]);

    return { connect, disconnect, isConnected, error, connectionState, applyQuality };
}

export default useWebRTC;