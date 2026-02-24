import { useState, useCallback, useRef, useEffect } from 'react';

function useWebcam(withAudio = false, audioDelayMs = 300) {
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
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: withAudio
            });

            if (withAudio) {
                // Enumerate all devices and log them for debugging
                const devices = await navigator.mediaDevices.enumerateDevices();
                console.log('[VirtualMic] All audio devices:');
                devices.filter(d => d.kind.startsWith('audio')).forEach(d => {
                    console.log(`  [${d.kind}] ${d.label} (${d.deviceId})`);
                });

                // Look for BlackHole as an audio OUTPUT (for setSinkId routing)
                const virtualOutput = devices.find(device =>
                    device.kind === 'audiooutput' &&
                    (device.label.toLowerCase().includes('vb-audio') || device.label.toLowerCase().includes('blackhole'))
                );

                if (!virtualOutput) {
                    console.warn('[VirtualMic] BlackHole/VB-Audio not found as audiooutput. Audio delay for lip-sync will not be active.');
                } else {
                    // Set up Web Audio API to delay the mic
                    const audioCtx = new AudioContext();
                    audioCtxRef.current = audioCtx;
                    const source = audioCtx.createMediaStreamSource(mediaStream);

                    const delayNode = audioCtx.createDelay(2.0); // max delay 2s
                    delayNode.delayTime.value = audioDelayMs / 1000;
                    delayNodeRef.current = delayNode;

                    const destination = audioCtx.createMediaStreamDestination();

                    source.connect(delayNode);
                    delayNode.connect(destination);

                    // Create hidden audio element to pipe the delayed stream to the Virtual Mic
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
    }, [withAudio, audioDelayMs]);

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
