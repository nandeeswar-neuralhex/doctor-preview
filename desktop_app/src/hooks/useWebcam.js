import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * Adaptive webcam capture hook.
 *
 * Key improvements:
 * - Accepts captureWidth/captureHeight from hardware profile (no hardcoded 1280x720)
 * - Cross-platform virtual audio: detects BlackHole (macOS), VB-Audio (Windows),
 *   PulseAudio null sinks (Linux), and gracefully degrades if none found
 * - Audio delay node for lip-sync compensation
 */
function useWebcam(withAudio = false, audioDelayMs = 300, captureWidth = 1280, captureHeight = 720) {
    const [stream, setStream] = useState(null);
    const [error, setError] = useState(null);
    const virtualAudioRef = useRef(null);
    const audioCtxRef = useRef(null);
    const delayNodeRef = useRef(null);

    // Dynamically update the delay when the slider changes
    useEffect(() => {
        if (delayNodeRef.current) {
            delayNodeRef.current.delayTime.value = audioDelayMs / 1000;
            console.log(`[VirtualMic] Delay updated to ${audioDelayMs}ms`);
        }
    }, [audioDelayMs]);

    const startWebcam = useCallback(async () => {
        try {
            // Adaptive capture: request resolution matching hardware tier
            // cheap webcams that only do 480p won't be forced to upscale
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: captureWidth, max: 1920 },
                    height: { ideal: captureHeight, max: 1080 },
                    frameRate: { ideal: 30, max: 30 },
                    facingMode: 'user'
                },
                audio: withAudio
            });

            // Log actual capture resolution
            const vTrack = mediaStream.getVideoTracks()[0];
            if (vTrack) {
                const settings = vTrack.getSettings();
                console.log(`[Webcam] Capturing at ${settings.width}x${settings.height} @ ${settings.frameRate}fps`);
            }

            if (withAudio) {
                const devices = await navigator.mediaDevices.enumerateDevices();
                console.log('[VirtualMic] All audio devices:');
                devices.filter(d => d.kind.startsWith('audio')).forEach(d => {
                    console.log(`  [${d.kind}] ${d.label} (${d.deviceId})`);
                });

                // Cross-platform virtual audio output detection
                const virtualOutput = devices.find(device => {
                    if (device.kind !== 'audiooutput') return false;
                    const label = device.label.toLowerCase();
                    return (
                        label.includes('blackhole') ||        // macOS
                        label.includes('vb-audio') ||         // Windows (VB-Cable)
                        label.includes('cable input') ||      // Windows (VB-Cable alternative name)
                        label.includes('virtual') ||          // Generic virtual audio devices
                        label.includes('null output') ||      // Linux PulseAudio
                        label.includes('loopback')            // Various loopback drivers
                    );
                });

                if (!virtualOutput) {
                    console.warn('[VirtualMic] No virtual audio output found. Audio delay inactive.');
                    console.warn('  macOS: Install BlackHole (brew install blackhole-2ch)');
                    console.warn('  Windows: Install VB-Audio Virtual Cable (https://vb-audio.com/Cable/)');
                } else {
                    // Set up Web Audio API to delay the mic for lip-sync
                    const audioCtx = new AudioContext();
                    audioCtxRef.current = audioCtx;
                    const source = audioCtx.createMediaStreamSource(mediaStream);

                    const delayNode = audioCtx.createDelay(2.0);
                    delayNode.delayTime.value = audioDelayMs / 1000;
                    delayNodeRef.current = delayNode;

                    const destination = audioCtx.createMediaStreamDestination();

                    source.connect(delayNode);
                    delayNode.connect(destination);

                    const audioEl = new Audio();
                    audioEl.srcObject = destination.stream;
                    audioEl.muted = true;

                    try {
                        if (typeof audioEl.setSinkId === 'function') {
                            await audioEl.setSinkId(virtualOutput.deviceId);
                            audioEl.muted = false;
                            audioEl.play().catch(e => console.error("[VirtualMic] Error playing audio:", e));
                            virtualAudioRef.current = audioEl;
                            console.log(`[VirtualMic] SUCCESS! Routed ${audioDelayMs}ms delayed audio to: ${virtualOutput.label}`);
                        } else {
                            console.warn("[VirtualMic] setSinkId is not supported in this environment");
                        }
                    } catch (sinkErr) {
                        console.error("[VirtualMic] Failed to route audio to virtual mic:", sinkErr);
                    }
                }
            }

            setStream(mediaStream);
            setError(null);
            return mediaStream;
        } catch (err) {
            setError(`Failed to access webcam: ${err.message}`);
            console.error('Webcam error:', err);
            return null;
        }
    }, [withAudio, audioDelayMs, captureWidth, captureHeight]);

    const stopWebcam = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
        if (virtualAudioRef.current) {
            virtualAudioRef.current.pause();
            virtualAudioRef.current.srcObject = null;
            virtualAudioRef.current = null;
        }
        if (audioCtxRef.current) {
            audioCtxRef.current.close().catch(console.error);
            audioCtxRef.current = null;
        }
        delayNodeRef.current = null;
    }, [stream]);

    return { stream, error, startWebcam, stopWebcam };
}

export default useWebcam;
