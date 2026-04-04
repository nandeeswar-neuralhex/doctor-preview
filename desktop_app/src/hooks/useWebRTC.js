import { useCallback, useEffect, useRef, useState } from 'react';
import QUALITY_PRESETS from '../qualityPresets';

function useWebRTC(serverUrl, sessionId, onRemoteStream, onLatency) {
    const pcRef = useRef(null);
    const dcRef = useRef(null);
    const pingIntervalRef = useRef(null);
    const latencyPollRef = useRef(null);
    const onLatencyRef = useRef(onLatency);
    const prevStatsRef = useRef({
        jitterBufferDelay: 0,
        jitterBufferEmittedCount: 0,
        totalPacketSendDelay: 0,
        packetsSent: 0,
        lastFrameTime: 0,
        lastBaseLatency: 0,
    });
    const [error, setError] = useState(null);
    const [isConnected, setIsConnected] = useState(false);
    const [connectionState, setConnectionState] = useState('new');

    useEffect(() => {
        onLatencyRef.current = onLatency;
    }, [onLatency]);

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

            // Close any existing connection before creating a new one
            // (safe reconnect — avoids leaked peer connections)
            if (pcRef.current) {
                try { pcRef.current.close(); } catch (_) {}
                pcRef.current = null;
            }

            const pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });

            const dc = pc.createDataChannel('latency', { ordered: false, maxRetransmits: 0 });
            dc.onclose = () => {
                if (pingIntervalRef.current) {
                    clearInterval(pingIntervalRef.current);
                    pingIntervalRef.current = null;
                }
            };
            dcRef.current = dc;

            prevStatsRef.current = {
                jitterBufferDelay: 0,
                jitterBufferEmittedCount: 0,
                totalPacketSendDelay: 0,
                packetsSent: 0,
                lastFrameTime: Date.now(),
                lastBaseLatency: 0,
            };
            latencyPollRef.current = setInterval(async () => {
                try {
                    const stats = await pc.getStats();
                    let iceRttMs = 0;
                    let jbDelayMs = 0;
                    let sendDelayMs = 0;
                    let inboundHasNewFrames = false;

                    for (const report of stats.values()) {
                        if (report.type === 'candidate-pair' && report.state === 'succeeded' && report.currentRoundTripTime != null) {
                            iceRttMs = report.currentRoundTripTime * 1000;
                        }

                        if (report.type === 'outbound-rtp' && report.kind === 'video' && report.totalPacketSendDelay != null) {
                            const prev = prevStatsRef.current;
                            const deltaSendDelay = report.totalPacketSendDelay - prev.totalPacketSendDelay;
                            const deltaPkts = report.packetsSent - prev.packetsSent;
                            if (deltaPkts > 0) {
                                sendDelayMs = (deltaSendDelay / deltaPkts) * 1000;
                            }
                            prevStatsRef.current.totalPacketSendDelay = report.totalPacketSendDelay;
                            prevStatsRef.current.packetsSent = report.packetsSent;
                        }

                        if (report.type === 'inbound-rtp' && report.kind === 'video') {
                            const prev = prevStatsRef.current;
                            const deltaDelay = report.jitterBufferDelay - prev.jitterBufferDelay;
                            const deltaCount = report.jitterBufferEmittedCount - prev.jitterBufferEmittedCount;
                            if (deltaCount > 0) {
                                jbDelayMs = (deltaDelay / deltaCount) * 1000;
                                inboundHasNewFrames = true;
                            }
                            prevStatsRef.current.jitterBufferDelay = report.jitterBufferDelay;
                            prevStatsRef.current.jitterBufferEmittedCount = report.jitterBufferEmittedCount;
                        }
                    }

                    if (inboundHasNewFrames) {
                        const totalLatency = Math.round(sendDelayMs + iceRttMs + 75 + jbDelayMs);
                        prevStatsRef.current.lastFrameTime = Date.now();
                        prevStatsRef.current.lastBaseLatency = totalLatency;
                        if (onLatencyRef.current) {
                            onLatencyRef.current(totalLatency);
                        }
                    } else {
                        const stallMs = Date.now() - prevStatsRef.current.lastFrameTime;
                        const totalLatency = Math.round(prevStatsRef.current.lastBaseLatency + stallMs);
                        if (onLatencyRef.current) {
                            onLatencyRef.current(totalLatency);
                        }
                    }
                } catch (_) {
                }
            }, 1000);

            localStream.getTracks().forEach(track => pc.addTrack(track, localStream));

            // Collect all remote tracks into one MediaStream so audio+video
            // are on the same stream object for the <video> element.
            const combinedStream = new MediaStream();

            pc.ontrack = (event) => {
                combinedStream.addTrack(event.track);
                console.log(`[WebRTC] Got remote ${event.track.kind} track (total: ${combinedStream.getTracks().length})`);
                if (onRemoteStream) {
                    onRemoteStream(combinedStream);
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
        if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
        }
        if (latencyPollRef.current) {
            clearInterval(latencyPollRef.current);
            latencyPollRef.current = null;
        }
        if (dcRef.current) {
            try {
                dcRef.current.close();
            } catch (_) {
            }
            dcRef.current = null;
        }
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