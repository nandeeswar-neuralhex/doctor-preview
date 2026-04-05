import React, { useRef, useEffect, useState, useCallback } from 'react';
import useWebcam from '../hooks/useWebcam';
import useWebRTC from '../hooks/useWebRTC';
import useWebSocket from '../hooks/useWebSocket';
import useWorkerTimer from '../hooks/useWorkerTimer';
import QualitySelector from './QualitySelector';
import QUALITY_PRESETS, { DEFAULT_QUALITY } from '../qualityPresets';

function CameraView({ serverUrl, targetImage, allTargetImages, isStreaming, setIsStreaming }) {
    const originalVideoRef = useRef(null);
    const processedVideoRef = useRef(null);
    const wsCanvasRef = useRef(null);  // Direct canvas paint — zero flicker
    const wsFrameCountRef = useRef(0);
    const wsLastFpsTimeRef = useRef(performance.now());
    const audioCtxRef = useRef(null);
    const audioBufferRef = useRef(new Int16Array(0));
    const audioSampleRateRef = useRef(16000);

    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [transportMode, setTransportMode] = useState('webrtc'); // 'webrtc' | 'websocket'
    const [qualityPreset, setQualityPreset] = useState(DEFAULT_QUALITY);
    const qualityRef = useRef(QUALITY_PRESETS[DEFAULT_QUALITY]);
    const [lipSyncEnabled, setLipSyncEnabled] = useState(true);
    const [autoSyncAudio, setAutoSyncAudio] = useState(true);
    const [extraAudioDelayMs, setExtraAudioDelayMs] = useState(0);
    const audioDelayMs = autoSyncAudio ? latency + extraAudioDelayMs : extraAudioDelayMs;
    const [exposureAdjust, setExposureAdjust] = useState(0);
    const [diagnostics, setDiagnostics] = useState({
        health: null,
        upload: null,
        settings: null,
        webrtc: null,
        localMedia: null,
        remoteMedia: null
    });
    const [fullScreenView, setFullScreenView] = useState(null); // 'original' | 'processed' | null

    // Function to mask MJPEG URL - shows only last 2 digits of IP and session timestamp
    const getMaskedMjpegUrl = (url) => {
        if (!url) return '...';
        // Extract last 2 digits from IP and show only last 6 digits of session ID
        const ipMatch = url.match(/(\d+)\.(\d+)\.(\d+)\.(\d{2,3})/);
        const sessionMatch = url.match(/session-(\d+)/);
        if (ipMatch && sessionMatch) {
            const lastDigits = ipMatch[4].slice(-2);
            const sessionDigits = sessionMatch[1].slice(-6);
            return `...${lastDigits}/mjpeg/...${sessionDigits}`;
        }
        return '...';
    };

    const toggleFullScreen = (view) => {
        setFullScreenView(prev => prev === view ? null : view);
    };

    const [isPoppedOut, setIsPoppedOut] = useState(false);
    const popoutWindowRef = useRef(null);
    const broadcastRef = useRef(null);

    // Open/close a regular resizable popup window that the user can place anywhere.
    // Frames are pushed via BroadcastChannel — the popup renders them on its own canvas.
    // Unlike PiP this does NOT float on top of everything.
    const handlePopOut = () => {
        // Close if already open
        if (popoutWindowRef.current && !popoutWindowRef.current.closed) {
            popoutWindowRef.current.close();
            popoutWindowRef.current = null;
            setIsPoppedOut(false);
            return;
        }

        const w = window.open(
            '',
            'doctor-preview-output',
            'width=720,height=540,resizable=yes,scrollbars=no,toolbar=no,menubar=no,location=no,status=no'
        );
        if (!w) {
            alert('Pop-out blocked. Please allow pop-ups for this site in your browser settings.');
            return;
        }

        // Write a self-contained canvas page into the popup
        w.document.write(`<!DOCTYPE html>
<html><head><title>AI Preview</title><style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#000; display:flex; align-items:center; justify-content:center; width:100vw; height:100vh; overflow:hidden; }
canvas { max-width:100%; max-height:100%; }
</style></head>
<body><canvas id="c"></canvas><script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
const bc = new BroadcastChannel('doctor-preview-frames');
bc.onmessage = (e) => {
  createImageBitmap(e.data).then(bmp => {
    if (canvas.width !== bmp.width) canvas.width = bmp.width;
    if (canvas.height !== bmp.height) canvas.height = bmp.height;
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
  });
};
window.addEventListener('beforeunload', () => bc.close());
</script></body></html>`);
        w.document.close();

        popoutWindowRef.current = w;
        setIsPoppedOut(true);

        // Detect when user closes the popup manually
        const pollClose = setInterval(() => {
            if (w.closed) {
                clearInterval(pollClose);
                popoutWindowRef.current = null;
                setIsPoppedOut(false);
            }
        }, 500);
    };

    const processedFrameFilter = `brightness(${Math.max(0.4, 1 + exposureAdjust / 100)})`;

    // Custom hooks for webcam and WebSocket
    // In WebRTC mode, skip the virtual audio delay pipeline — the server
    // relays audio back via a simple passthrough, and Chrome's built-in A/V
    // sync (RTCP Sender Reports) holds audio to match the ~130ms video delay.
    // The processed <video> element's audio is routed to BlackHole via setSinkId.
    const skipVirtualAudio = transportMode === 'webrtc';
    const { stream, error: webcamError, startWebcam, stopWebcam } = useWebcam(true, audioDelayMs, skipVirtualAudio);
    // WebSocket hook – render into the dedicated <img> ref
    const handleWsFrame = useCallback((frameData, wsLatency, isBinary) => {
        // Direct canvas painting: decode blob → drawImage → done.
        // createImageBitmap() decodes off main thread → zero jank.
        // No DOM swaps, no opacity transitions, no blob URLs = zero blink.
        const canvas = wsCanvasRef.current;
        if (!canvas) return;

        if (isBinary && frameData instanceof Blob) {
            // Binary mode: raw Blob from WebSocket → decode → paint
            createImageBitmap(frameData)
                .then(bitmap => {
                    const ctx = canvas.getContext('2d');
                    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
                        canvas.width = bitmap.width;
                        canvas.height = bitmap.height;
                        // Log response resolution once when it changes
                        console.log(`[WS-RECV] Server response: ${bitmap.width}×${bitmap.height}`);
                        setDiagnostics(prev => ({ ...prev, remoteMedia: `Receiving ${bitmap.width}×${bitmap.height}` }));
                    }
                    ctx.drawImage(bitmap, 0, 0);
                    bitmap.close();
                    // Mirror to popout window via BroadcastChannel
                    if (broadcastRef.current) broadcastRef.current.postMessage(frameData);
                })
                .catch((err) => {
                    console.error('Canvas paint error:', err);
                    setDiagnostics(prev => ({ ...prev, remoteMedia: `Paint error: ${err.message}` }));
                });
        } else if (typeof frameData === 'string') {
            // Legacy text mode: data: URI
            const img = new Image();
            img.onload = () => {
                const ctx = canvas.getContext('2d');
                if (canvas.width !== img.width || canvas.height !== img.height) {
                    canvas.width = img.width;
                    canvas.height = img.height;
                }
                ctx.drawImage(img, 0, 0);
            };
            img.src = frameData;
        }
        // FPS counter for WS mode
        wsFrameCountRef.current++;
        const now = performance.now();
        if (now - wsLastFpsTimeRef.current >= 1000) {
            setFps(wsFrameCountRef.current);
            wsFrameCountRef.current = 0;
            wsLastFpsTimeRef.current = now;
        }
        // Update latency if provided
        if (wsLatency !== undefined) {
            setLatency(wsLatency);
        }
    }, []);

    const {
        connect: connectWs,
        disconnect: disconnectWs,
        sendFrame: sendWsFrame,
        isConnected: isWsConnected,
        error: wsError
    } = useWebSocket(serverUrl, sessionId, handleWsFrame);

    const remoteStreamRef = useRef(null);

    const {
        isConnected,
        error: rtcError,
        connectionState,
        connect,
        disconnect,
        applyQuality
    } = useWebRTC(serverUrl, sessionId, (remoteStream) => {
        remoteStreamRef.current = remoteStream;
        // NOTE: processedVideoRef.current is usually null here because
        // ontrack fires before isStreaming=true renders the <video>.
        // The useEffect below handles attaching + BlackHole routing.
        if (processedVideoRef.current) {
            processedVideoRef.current.srcObject = remoteStream;
        }
        setDiagnostics(prev => ({
            ...prev,
            remoteMedia: {
                videoTracks: remoteStream.getVideoTracks().length,
                audioTracks: remoteStream.getAudioTracks().length
            }
        }));
    }, (rtt) => {
        setLatency(rtt);
    });

    // Track whether we successfully routed audio to BlackHole
    const audioRoutedToBlackHoleRef = useRef(false);

    // Attach remote stream to video element once it renders, and route
    // audio to BlackHole. Two race conditions to handle:
    // 1. ontrack fires before <video> exists — useEffect waits for isStreaming+isConnected
    // 2. Audio ontrack fires AFTER video ontrack. When isConnected first fires,
    //    the stream may only have video. Adding remoteMedia.audioTracks to deps
    //    re-runs this effect the moment the audio track arrives.
    useEffect(() => {
        if (!isStreaming || !isConnected || !processedVideoRef.current || !remoteStreamRef.current) return;

        const videoEl = processedVideoRef.current;
        videoEl.srcObject = remoteStreamRef.current;

        // Route audio to BlackHole — only possible after audio track has arrived
        const audioTracks = remoteStreamRef.current.getAudioTracks();
        if (audioTracks.length === 0) return;  // will re-run when audio track arrives

        let cancelled = false;
        (async () => {
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                if (cancelled) return;
                const virtualOutput = devices.find(d =>
                    d.kind === 'audiooutput' &&
                    (d.label.toLowerCase().includes('blackhole') ||
                     d.label.toLowerCase().includes('vb-audio'))
                );
                if (virtualOutput && typeof videoEl.setSinkId === 'function') {
                    await videoEl.setSinkId(virtualOutput.deviceId);
                    if (cancelled) return;
                    // Sink is set to BlackHole — safe to un-mute and ensure playing
                    videoEl.muted = false;
                    videoEl.play().catch(() => {});
                    audioRoutedToBlackHoleRef.current = true;
                    console.log(`[WebRTC] Audio routed to BlackHole: ${virtualOutput.label}`);
                } else {
                    console.warn('[WebRTC] BlackHole/VB-Audio not found — audio stays muted');
                    audioRoutedToBlackHoleRef.current = false;
                }
            } catch (e) {
                console.warn('[WebRTC] Failed to route audio:', e.message);
                audioRoutedToBlackHoleRef.current = false;
            }
        })();

        return () => { cancelled = true; };
    }, [isStreaming, isConnected, diagnostics.remoteMedia?.audioTracks]);

    // Fallback logic: If WebRTC fails or disconnects, try WebSocket
    useEffect(() => {
        if (isStreaming && !isConnected && connectionState === 'failed' && !isWsConnected) {
            console.log('WebRTC failed, falling back to WebSocket...');
            setDiagnostics(prev => ({ ...prev, webrtc: 'WebRTC failed, trying WebSocket...' }));
            connectWs();
        }
    }, [isStreaming, isConnected, connectionState, isWsConnected, connectWs]);

    // Web Worker timer — immune to browser background-tab throttling.
    // Browsers clamp setTimeout to ≥1 s for hidden tabs; Worker threads keep
    // their own event-loop at full speed, so FPS stays steady even when the
    // doctor switches to another tab or app.
    const workerTimer = useWorkerTimer();

    // Background-mode debug logger — logs once per second so we can verify
    // ticks are arriving even when the tab is hidden. Open DevTools Console
    // then switch tabs to check.
    const bgDebugRef = useRef({ ticks: 0, sends: 0, errors: 0, lastLog: 0 });

    // Send frames via WebSocket if connected – throttled by quality preset FPS
    useEffect(() => {
        if (!isStreaming || !isWsConnected || !originalVideoRef.current) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const video = originalVideoRef.current;

        // Grab the video track directly from the camera — ImageCapture reads from
        // the hardware driver, NOT the <video> element. This keeps working when
        // the browser tab is hidden (where drawImage(video, …) returns a black
        // frame because Chromium pauses video rendering).
        const videoTrack = video.srcObject?.getVideoTracks()[0];
        const imageCapture = videoTrack ? new ImageCapture(videoTrack) : null;

        console.log('[BG-DEBUG] Send loop started.', {
            hasImageCapture: !!imageCapture,
            hasVideoTrack: !!videoTrack,
            videoReadyState: video.readyState,
            hidden: document.hidden,
        });

        let active = true;
        let grabInFlight = false;
        let loggedResolution = false;

        const getQuality = () => qualityRef.current;

        const grabAndSend = (bitmap) => {
            const q = getQuality();
            // Downscale to match preset (never upscale — camera should match)
            const scale = Math.min(1, q.width / bitmap.width);
            canvas.width = Math.round(bitmap.width * scale);
            canvas.height = Math.round(bitmap.height * scale);
            if (!loggedResolution) {
                loggedResolution = true;
                console.log(`[WS-SEND] Camera=${bitmap.width}×${bitmap.height} → Sending=${canvas.width}×${canvas.height} (preset=${q.width}×${q.height}, jpegQ=${q.jpegQuality})`);
            }
            ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
            bitmap.close();
            canvas.toBlob((blob) => {
                if (!active || !blob) return;
                bgDebugRef.current.sends++;
                sendWsFrame(blob, audioBufferRef.current, audioSampleRateRef.current);
            }, 'image/jpeg', q.jpegQuality);
        };

        const onTick = () => {
            if (!active || grabInFlight) return;

            bgDebugRef.current.ticks++;
            // Log once per second so we can see the ticker in DevTools even when backgrounded
            const now = Date.now();
            if (now - bgDebugRef.current.lastLog > 1000) {
                console.log(`[BG-DEBUG] ticks=${bgDebugRef.current.ticks} sends=${bgDebugRef.current.sends} errors=${bgDebugRef.current.errors} hidden=${document.hidden}`);
                bgDebugRef.current.lastLog = now;
            }

            grabInFlight = true;

            if (imageCapture) {
                imageCapture.grabFrame()
                    .then(bitmap => { if (active) grabAndSend(bitmap); })
                    .catch((err) => {
                        bgDebugRef.current.errors++;
                        if (active && video.readyState === video.HAVE_ENOUGH_DATA) {
                            const q2 = getQuality();
                            const scale = Math.min(1, q2.width / video.videoWidth);
                            canvas.width = Math.round(video.videoWidth * scale);
                            canvas.height = Math.round(video.videoHeight * scale);
                            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                            canvas.toBlob((blob) => {
                                if (!active || !blob) return;
                                bgDebugRef.current.sends++;
                                sendWsFrame(blob, audioBufferRef.current, audioSampleRateRef.current);
                            }, 'image/jpeg', q2.jpegQuality);
                        }
                    })
                    .finally(() => { grabInFlight = false; });
            } else {
                if (video.readyState === video.HAVE_ENOUGH_DATA) {
                    const q2 = getQuality();
                    const scale = Math.min(1, q2.width / video.videoWidth);
                    canvas.width = Math.round(video.videoWidth * scale);
                    canvas.height = Math.round(video.videoHeight * scale);
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    canvas.toBlob((blob) => {
                        if (!active || !blob) return;
                        bgDebugRef.current.sends++;
                        sendWsFrame(blob, audioBufferRef.current, audioSampleRateRef.current);
                    }, 'image/jpeg', q2.jpegQuality);
                }
                grabInFlight = false;
            }
        };

        // Start the Worker-based interval — NOT throttled in background tabs.
        // Combined with Web Lock (inside useWorkerTimer), Chrome won't freeze us.
        const intervalMs = Math.round(1000 / getQuality().fps);
        workerTimer.start(intervalMs, onTick);

        return () => {
            active = false;
            workerTimer.stop();
            console.log('[BG-DEBUG] Send loop stopped. Final stats:', { ...bgDebugRef.current });
            bgDebugRef.current = { ticks: 0, sends: 0, errors: 0, lastLog: 0 };
        };
    }, [isStreaming, isWsConnected, sendWsFrame, workerTimer]);

    // ── Background Keepalive ──
    // Prevents macOS App Nap + Chrome page freeze when switching applications.
    // Two layers:
    //  1. Web Lock — tells Chrome this page has critical background work
    //  2. Silent audio oscillator — prevents macOS from napping the Chrome process
    const keepaliveLockRef = useRef(null);
    const keepaliveAudioRef = useRef(null);
    useEffect(() => {
        if (!isStreaming) return;

        // 1. Acquire Web Lock (prevents Chrome page freeze after 5 min)
        let lockAc = null;
        let lockResolve = null;
        if (navigator.locks) {
            lockAc = new AbortController();
            navigator.locks.request(
                'doctor-preview-stream-keepalive',
                { signal: lockAc.signal },
                () => new Promise((resolve) => { lockResolve = resolve; })
            ).catch(() => {}); // AbortError on release — expected
            console.log('[Keepalive] Web Lock acquired');
        }
        keepaliveLockRef.current = { lockAc, lockResolve };

        // 2. Silent audio oscillator (prevents macOS App Nap)
        let silentCtx = null;
        try {
            silentCtx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = silentCtx.createOscillator();
            const gain = silentCtx.createGain();
            osc.frequency.value = 1; // 1 Hz — inaudible
            gain.gain.value = 0.001; // essentially silent
            osc.connect(gain);
            gain.connect(silentCtx.destination);
            osc.start();
            console.log('[Keepalive] Silent audio oscillator started');
            keepaliveAudioRef.current = { ctx: silentCtx, osc, gain };
        } catch (e) {
            console.warn('[Keepalive] Silent audio failed:', e.message);
        }

        return () => {
            // Release Web Lock
            if (lockResolve) lockResolve();
            if (lockAc) lockAc.abort();
            keepaliveLockRef.current = null;
            console.log('[Keepalive] Web Lock released');
            // Stop silent audio
            if (keepaliveAudioRef.current) {
                try { keepaliveAudioRef.current.osc.stop(); } catch (_) {}
                try { keepaliveAudioRef.current.ctx.close(); } catch (_) {}
                keepaliveAudioRef.current = null;
            }

        };
    }, [isStreaming]);

    // Keep quality ref in sync and apply live to WebRTC senders
    const handleQualityChange = useCallback((newPreset) => {
        setQualityPreset(newPreset);
        qualityRef.current = QUALITY_PRESETS[newPreset];
        // Update Worker timer interval to match new FPS
        const preset = QUALITY_PRESETS[newPreset];
        if (preset) workerTimer.setInterval(Math.round(1000 / preset.fps));
        // Apply to WebRTC senders if connected
        if (isConnected) {
            applyQuality(newPreset);
        }
    }, [isConnected, applyQuality, workerTimer]);

    // Auto-reconnect WebRTC when the connection drops while streaming.
    // aiortc closes the connection when DTLS/ICE keepalives stop (browser
    // suspends them when the tab/app goes to background on macOS).
    const reconnectTimerRef = useRef(null);
    useEffect(() => {
        if (!isStreaming || transportMode !== 'webrtc') return;
        // Only reconnect when transitioning to a dead state
        if (!['disconnected', 'failed', 'closed'].includes(connectionState)) return;
        if (!stream) return;

        console.log(`[WebRTC-RECONNECT] Connection ${connectionState} while streaming — will reconnect in 1.5s`);
        reconnectTimerRef.current = setTimeout(async () => {
            try {
                console.log('[WebRTC-RECONNECT] Reconnecting...');
                await connect(stream, qualityPreset);
                console.log('[WebRTC-RECONNECT] Reconnected!');
            } catch (err) {
                console.error('[WebRTC-RECONNECT] Failed:', err.message);
            }
        }, 1500);

        return () => {
            if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
        };
    }, [isStreaming, connectionState, transportMode, stream, qualityPreset, connect]);

    // Keep video/audio alive when the browser tab goes to background.
    // Browsers suspend AudioContext and may pause video tracks—resume them
    // immediately when the page becomes visible again. Also reconnect WebRTC
    // if the connection died while backgrounded.
    useEffect(() => {
        if (!isStreaming) return;

        const onVisibilityChange = () => {
            if (document.visibilityState === 'visible') {
                // Resume AudioContext if browser suspended it
                if (audioCtxRef.current && audioCtxRef.current.state === 'suspended') {
                    audioCtxRef.current.resume().catch(() => {});
                }
                // Re-attach remote stream to wake up the frozen <video> element
                if (processedVideoRef.current && remoteStreamRef.current && isConnected) {
                    processedVideoRef.current.srcObject = remoteStreamRef.current;
                    // Re-assert un-mute if BlackHole routing was already done
                    if (audioRoutedToBlackHoleRef.current) {
                        processedVideoRef.current.muted = false;
                    }
                    processedVideoRef.current.play().catch(() => {});
                }
                // Ensure camera track is still enabled
                if (originalVideoRef.current?.srcObject) {
                    originalVideoRef.current.srcObject.getVideoTracks().forEach(t => { t.enabled = true; });
                }
                // If WebRTC died while in background, reconnect immediately
                if (transportMode === 'webrtc' && !isConnected && stream) {
                    console.log('[WebRTC-RECONNECT] Visibility restored, connection dead — reconnecting');
                    connect(stream, qualityPreset).catch((err) => {
                        console.error('[WebRTC-RECONNECT] Reconnect on visibility failed:', err.message);
                    });
                }
            }
        };

        document.addEventListener('visibilitychange', onVisibilityChange);
        return () => document.removeEventListener('visibilitychange', onVisibilityChange);
    }, [isStreaming, isConnected, transportMode, stream, qualityPreset, connect]);

    // Auto re-upload ALL target images when the user adds/removes images mid-stream
    // Track previous images to prevent redundant upload on start (when isStreaming flips to true)
    const prevImagesRef = useRef(allTargetImages);

    useEffect(() => {
        if (!isStreaming || !serverUrl || !allTargetImages || allTargetImages.length === 0) return;

        // Skip if images haven't changed since last successful upload (e.g. initial start)
        // We compare length or reference. For deeper check, we rely on parent to maintain stable references.
        if (prevImagesRef.current === allTargetImages) return;
        prevImagesRef.current = allTargetImages;

        const reupload = async () => {
            try {
                if (allTargetImages.length > 1) {
                    setDiagnostics(prev => ({ ...prev, upload: `Re-uploading ${allTargetImages.length} images...` }));
                    const formData = new FormData();
                    allTargetImages.forEach(img => {
                        formData.append('files', img.file);
                    });
                    const resp = await fetch(
                        `${serverUrl}/upload-targets?session_id=${sessionId}`,
                        { method: 'POST', body: formData }
                    );
                    const body = await resp.json().catch(() => null);
                    if (!resp.ok) {
                        setDiagnostics(prev => ({
                            ...prev,
                            upload: `Re-upload failed: ${body?.error || resp.status}`
                        }));
                        return;
                    }
                    setDiagnostics(prev => ({ ...prev, upload: `Re-upload ok: ${body?.message || 'success'}` }));
                } else {
                    setDiagnostics(prev => ({ ...prev, upload: 'Re-uploading target image...' }));
                    const formData = new FormData();
                    formData.append('file', allTargetImages[0].file);
                    const resp = await fetch(
                        `${serverUrl}/upload-target?session_id=${sessionId}`,
                        { method: 'POST', body: formData }
                    );
                    const body = await resp.json().catch(() => null);
                    if (!resp.ok) {
                        setDiagnostics(prev => ({
                            ...prev,
                            upload: `Re-upload failed: ${body?.error || resp.status}`
                        }));
                        return;
                    }
                    setDiagnostics(prev => ({ ...prev, upload: `Re-upload ok: ${body?.message || 'success'}` }));
                }
            } catch (err) {
                setDiagnostics(prev => ({ ...prev, upload: `Re-upload error: ${err.message}` }));
            }
        };
        reupload();
    }, [allTargetImages, isStreaming, serverUrl, sessionId]);

    // Display webcam stream in video element
    useEffect(() => {
        if (stream && originalVideoRef.current) {
            originalVideoRef.current.srcObject = stream;
            setDiagnostics(prev => ({
                ...prev,
                localMedia: {
                    videoTracks: stream.getVideoTracks().length,
                    audioTracks: stream.getAudioTracks().length
                }
            }));
        }
    }, [stream]);

    // Capture mic audio for lip sync when using WebSocket mode
    useEffect(() => {
        if (!isStreaming || !isWsConnected || !stream) return;

        const audioTracks = stream.getAudioTracks();
        if (audioTracks.length === 0) {
            console.log('No audio tracks available for lip sync');
            return;
        }

        let disposed = false;
        let audioCtx;
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        } catch (e) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        audioSampleRateRef.current = audioCtx.sampleRate;
        audioCtxRef.current = audioCtx;

        const audioStream = new MediaStream(audioTracks);
        const source = audioCtx.createMediaStreamSource(audioStream);

        // Use AudioWorkletNode (runs off-main-thread, doesn't crash Electron 28).
        // ScriptProcessorNode causes a renderer SIGSEGV crash in Chromium 120.
        const setupAudioCapture = async () => {
            try {
                // Register an inline AudioWorklet processor via Blob URL
                const workletCode = `
                    class PcmCapture extends AudioWorkletProcessor {
                        process(inputs) {
                            const input = inputs[0];
                            if (input && input[0] && input[0].length > 0) {
                                this.port.postMessage(input[0]);
                            }
                            return true;
                        }
                    }
                    registerProcessor('pcm-capture', PcmCapture);
                `;
                const blob = new Blob([workletCode], { type: 'application/javascript' });
                const url = URL.createObjectURL(blob);
                await audioCtx.audioWorklet.addModule(url);
                URL.revokeObjectURL(url);

                if (disposed) return;

                const workletNode = new AudioWorkletNode(audioCtx, 'pcm-capture');
                workletNode.port.onmessage = (e) => {
                    if (disposed) return;
                    const float32 = e.data;
                    const int16 = new Int16Array(float32.length);
                    for (let i = 0; i < float32.length; i++) {
                        int16[i] = (float32[i] * 0x7FFF) | 0;
                    }
                    const maxSamples = 8000;
                    const prev = audioBufferRef.current;
                    if (prev.length === 0) {
                        audioBufferRef.current = int16.length > maxSamples
                            ? int16.slice(int16.length - maxSamples) : int16;
                    } else {
                        const combined = new Int16Array(prev.length + int16.length);
                        combined.set(prev);
                        combined.set(int16, prev.length);
                        audioBufferRef.current = combined.length > maxSamples
                            ? combined.slice(combined.length - maxSamples) : combined;
                    }
                };

                source.connect(workletNode);
                workletNode.connect(audioCtx.destination);
                console.log(`Audio capture started (AudioWorklet): ${audioCtx.sampleRate}Hz`);
            } catch (workletErr) {
                // AudioWorklet not supported — use AnalyserNode polling as safe fallback
                // (NOT ScriptProcessorNode which crashes Electron 28)
                console.warn('AudioWorklet failed, using AnalyserNode fallback:', workletErr.message);
                const analyser = audioCtx.createAnalyser();
                analyser.fftSize = 2048;
                source.connect(analyser);

                const captureInterval = setInterval(() => {
                    if (disposed) return;
                    const float32 = new Float32Array(analyser.fftSize);
                    analyser.getFloatTimeDomainData(float32);
                    const int16 = new Int16Array(float32.length);
                    for (let i = 0; i < float32.length; i++) {
                        int16[i] = (float32[i] * 0x7FFF) | 0;
                    }
                    const maxSamples = 8000;
                    const prev = audioBufferRef.current;
                    const combined = new Int16Array(prev.length + int16.length);
                    combined.set(prev);
                    combined.set(int16, prev.length);
                    audioBufferRef.current = combined.length > maxSamples
                        ? combined.slice(combined.length - maxSamples) : combined;
                }, 100); // ~10 captures/sec

                // Store interval for cleanup
                audioCtx._captureInterval = captureInterval;
                console.log(`Audio capture started (AnalyserNode fallback): ${audioCtx.sampleRate}Hz`);
            }
        };

        setupAudioCapture();

        // Chromium auto-suspends AudioContext when the page loses visibility.
        // Resume it immediately whenever the window comes back into view/focus.
        const resumeAudioCtx = () => {
            if (document.visibilityState === 'visible' && audioCtx && audioCtx.state === 'suspended') {
                audioCtx.resume().catch(() => {});
            }
        };
        document.addEventListener('visibilitychange', resumeAudioCtx);

        return () => {
            document.removeEventListener('visibilitychange', resumeAudioCtx);
            disposed = true;
            if (audioCtx._captureInterval) clearInterval(audioCtx._captureInterval);
            try { source.disconnect(); } catch (_) { }
            try { audioCtx.close(); } catch (_) { }
            audioCtxRef.current = null;
            audioBufferRef.current = new Int16Array(0);
            console.log('Audio capture stopped');
        };
    }, [isStreaming, isWsConnected, stream]);

    // FPS from processed video – count actual decoded frames
    useEffect(() => {
        if (!isStreaming) return;

        // For WebRTC: count via the video element
        const video = processedVideoRef.current;
        // For WebSocket: FPS is counted in handleWsFrame, skip this effect
        if (isWsConnected || !video) return;

        let frameCount = 0;
        let lastTime = performance.now();
        let active = true;

        // Preferred: requestVideoFrameCallback (counts real decoded frames)
        if (video instanceof HTMLVideoElement && 'requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            const tick = (now) => {
                if (!active) return;
                frameCount++;
                if (now - lastTime >= 1000) {
                    setFps(frameCount);
                    frameCount = 0;
                    lastTime = now;
                }
                video.requestVideoFrameCallback(tick);
            };
            video.requestVideoFrameCallback(tick);
        } else {
            // Fallback: use timeupdate events (~4/sec) scaled to estimate
            const onTimeUpdate = () => {
                frameCount++;
                const now = performance.now();
                if (now - lastTime >= 1000) {
                    setFps(frameCount);
                    frameCount = 0;
                    lastTime = now;
                }
            };
            video.addEventListener('timeupdate', onTimeUpdate);
            return () => {
                active = false;
                video.removeEventListener('timeupdate', onTimeUpdate);
            };
        }

        return () => { active = false; };
    }, [isStreaming, isConnected, isWsConnected]);

    // Create/destroy BroadcastChannel while streaming
    useEffect(() => {
        if (!isStreaming) return;
        const bc = new BroadcastChannel('doctor-preview-frames');
        broadcastRef.current = bc;
        return () => {
            bc.close();
            broadcastRef.current = null;
        };
    }, [isStreaming]);

    // WebRTC mode: copy video frames → BroadcastChannel for the popout window
    useEffect(() => {
        if (!isStreaming || !isConnected || isWsConnected) return;
        const video = processedVideoRef.current;
        if (!video) return;

        const offscreen = document.createElement('canvas');
        const ctx = offscreen.getContext('2d');
        let active = true;

        const postFrame = () => {
            if (!active) return;
            const bc = broadcastRef.current;
            if (bc && !video.paused && video.videoWidth > 0) {
                offscreen.width = video.videoWidth;
                offscreen.height = video.videoHeight;
                ctx.drawImage(video, 0, 0);
                offscreen.toBlob((blob) => {
                    if (blob && broadcastRef.current) broadcastRef.current.postMessage(blob);
                }, 'image/jpeg', 0.85);
            }
            if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
                video.requestVideoFrameCallback(postFrame);
            }
        };

        if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
            video.requestVideoFrameCallback(postFrame);
        }
        return () => { active = false; };
    }, [isStreaming, isConnected, isWsConnected]);

    const handleStart = async () => {
        if (!serverUrl) {
            alert('Please set server URL in settings first');
            return;
        }

        if (!targetImage) {
            alert('Please upload a target image first');
            return;
        }

        try {
            // Upload ALL target images for expression matching
            const imagesToUpload = allTargetImages && allTargetImages.length > 0 ? allTargetImages : [targetImage];

            if (imagesToUpload.length > 1) {
                // Multi-image upload for expression matching
                setDiagnostics(prev => ({ ...prev, upload: `Uploading ${imagesToUpload.length} images for expression matching...` }));
                const formData = new FormData();
                imagesToUpload.forEach(img => {
                    formData.append('files', img.file);
                });

                const response = await fetch(`${serverUrl}/upload-targets?session_id=${sessionId}`, {
                    method: 'POST',
                    body: formData
                });

                const responseBody = await response.json().catch(() => null);
                if (!response.ok) {
                    const errorMessage = responseBody?.error || 'Failed to upload target images';
                    setDiagnostics(prev => ({
                        ...prev,
                        upload: `Upload failed: ${errorMessage} (HTTP ${response.status})`
                    }));
                    throw new Error(errorMessage);
                }
                setDiagnostics(prev => ({
                    ...prev,
                    upload: `Upload ok: ${responseBody?.message || 'success'} (${responseBody?.faces_stored || '?'} faces)`
                }));
            } else {
                // Single image upload
                setDiagnostics(prev => ({ ...prev, upload: 'Uploading target image...' }));
                const formData = new FormData();
                formData.append('file', targetImage.file);

                const response = await fetch(`${serverUrl}/upload-target?session_id=${sessionId}`, {
                    method: 'POST',
                    body: formData
                });

                const responseBody = await response.json().catch(() => null);
                if (!response.ok) {
                    const errorMessage = responseBody?.error || 'Failed to upload target image';
                    setDiagnostics(prev => ({
                        ...prev,
                        upload: `Upload failed: ${errorMessage} (HTTP ${response.status})`
                    }));
                    if ((responseBody?.error || '').toLowerCase().includes('no face detected')) {
                        throw new Error('No face detected. Use a clear frontal target photo (full face, no extreme crop).');
                    }
                    throw new Error(errorMessage);
                }
                setDiagnostics(prev => ({
                    ...prev,
                    upload: `Upload ok: ${responseBody?.message || 'success'}`
                }));
            }

            // Apply session settings (fire-and-forget, don't block start)
            fetch(`${serverUrl}/session/settings?session_id=${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable_lipsync: lipSyncEnabled })
            }).then(() => {
                setDiagnostics(prev => ({ ...prev, settings: 'Settings applied' }));
            }).catch(() => {
                setDiagnostics(prev => ({ ...prev, settings: 'Settings failed (non-critical)' }));
            });

            // Start webcam at the resolution matching the quality preset
            const preset = QUALITY_PRESETS[qualityPreset] || QUALITY_PRESETS['720p'];
            const mediaStream = await startWebcam(preset.width, preset.height);
            if (!mediaStream) {
                throw new Error('Unable to access webcam/microphone');
            }

            // Log actual camera resolution for diagnostics
            const camTrack = mediaStream.getVideoTracks()[0];
            if (camTrack) {
                const s = camTrack.getSettings();
                setDiagnostics(prev => ({
                    ...prev,
                    localMedia: {
                        videoTracks: mediaStream.getVideoTracks().length,
                        audioTracks: mediaStream.getAudioTracks().length,
                        resolution: `${s.width}×${s.height}`,
                    }
                }));
            }

            if (transportMode === 'webrtc') {
                try {
                    setDiagnostics(prev => ({ ...prev, webrtc: 'Trying WebRTC...' }));
                    await connect(mediaStream, qualityPreset);
                    setDiagnostics(prev => ({ ...prev, webrtc: 'WebRTC connected!' }));
                } catch (rtcErr) {
                    console.log('WebRTC failed, falling back to WebSocket:', rtcErr.message);
                    setDiagnostics(prev => ({ ...prev, webrtc: `WebRTC failed: ${rtcErr.message} → using WebSocket` }));
                    connectWs();
                }
            } else {
                setDiagnostics(prev => ({ ...prev, webrtc: 'Using WebSocket (manual)' }));
                connectWs();
            }
            setIsStreaming(true);
        } catch (error) {
            setDiagnostics(prev => ({ ...prev, webrtc: `Connection error: ${error.message}` }));
            alert(`Error: ${error.message}`);
        }
    };

    const handleStop = () => {
        setIsStreaming(false);  // Set first to stop send loops immediately
        disconnectWs();
        disconnect();
        stopWebcam();
        setFps(0);
        setLatency(0);
        audioBufferRef.current = new Int16Array(0);
        setDiagnostics({ health: null, upload: null, settings: null, webrtc: null, localMedia: null, remoteMedia: null });
    };

    const handleHealthCheck = async () => {
        if (!serverUrl) {
            setDiagnostics(prev => ({ ...prev, health: 'Server URL not set' }));
            return;
        }
        try {
            setDiagnostics(prev => ({ ...prev, health: 'Checking /health...' }));
            const response = await fetch(`${serverUrl}/health`);
            const body = await response.json().catch(() => null);
            if (!response.ok) {
                setDiagnostics(prev => ({
                    ...prev,
                    health: `Health failed: ${body?.error || 'error'} (HTTP ${response.status})`
                }));
                return;
            }
            setDiagnostics(prev => ({
                ...prev,
                health: `Health ok: model_loaded=${body?.model_loaded}, active_sessions=${body?.active_sessions}`
            }));
        } catch (err) {
            setDiagnostics(prev => ({ ...prev, health: `Health error: ${err.message}` }));
        }
    };

    return (
        <div className="h-full flex flex-col">
            {/* Controls */}
            <div className="mb-3 md:mb-4 flex flex-wrap items-center gap-3 md:gap-4">
                <div className="flex items-center gap-3">
                    {!isStreaming ? (
                        <button
                            onClick={handleStart}
                            disabled={!serverUrl || !targetImage}
                            className="px-4 py-2 md:px-6 md:py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2 text-sm md:text-base"
                        >
                            <svg className="w-4 h-4 md:w-5 md:h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                            </svg>
                            Start Preview
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            className="px-4 py-2 md:px-6 md:py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2 text-sm md:text-base"
                        >
                            <svg className="w-4 h-4 md:w-5 md:h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                            </svg>
                            Stop
                        </button>
                    )}
                    <button
                        onClick={handleHealthCheck}
                        className="px-3 py-2 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                    >
                        Health
                    </button>
                </div>

                {/* Transport mode selector */}
                {!isStreaming && (
                    <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">Transport:</span>
                        <select
                            value={transportMode}
                            onChange={(e) => setTransportMode(e.target.value)}
                            className="bg-gray-700 border border-gray-600 text-white text-xs rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="webrtc">⚡ WebRTC (low latency)</option>
                            <option value="websocket">🔌 WebSocket (stable)</option>
                        </select>
                    </div>
                )}
                {isStreaming && (
                    <div className="flex items-center gap-1.5">
                        <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-purple-400 animate-pulse' : isWsConnected ? 'bg-blue-400 animate-pulse' : 'bg-gray-500'}`}></span>
                        <span className="text-xs text-gray-300">{isConnected ? '⚡ WebRTC' : isWsConnected ? '🔌 WebSocket' : 'disconnected'}</span>
                    </div>
                )}

                {/* Stats (inline with controls on mobile) */}
                {isStreaming && (
                    <div className="flex flex-wrap gap-3 md:gap-6 text-xs md:text-sm">
                        <div className="flex items-center gap-1">
                            <span className="text-gray-400">FPS:</span>
                            <span className="font-mono text-green-400 font-semibold">{fps}</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="text-gray-400">Latency:</span>
                            <span className="font-mono text-blue-400 font-semibold">{latency}ms</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : isWsConnected ? 'bg-blue-500 animate-pulse' : 'bg-red-500'}`}></span>
                            <span className="text-gray-400">
                                {isConnected ? 'WebRTC' : isWsConnected ? 'WS' : 'Off'}
                            </span>
                        </div>
                    </div>
                )}

                {/* YouTube-style quality selector — works during stream */}
                <QualitySelector
                    quality={qualityPreset}
                    onChange={handleQualityChange}
                />
            </div>

            {/* Settings quick toggles */}
            <div className="mb-3 md:mb-4 flex flex-wrap items-center gap-3 md:gap-4">
                <label className="flex items-center gap-2 text-xs md:text-sm text-gray-300">
                    <input
                        type="checkbox"
                        checked={lipSyncEnabled}
                        onChange={(e) => setLipSyncEnabled(e.target.checked)}
                        disabled={isStreaming}
                    />
                    Lip Sync
                </label>
                <div className="flex items-center gap-2 text-xs md:text-sm text-gray-300">
                    <label className="flex items-center gap-1 whitespace-nowrap cursor-pointer">
                        <input
                            type="checkbox"
                            checked={autoSyncAudio}
                            onChange={(e) => setAutoSyncAudio(e.target.checked)}
                            className="accent-blue-500"
                        />
                        Auto Sync
                    </label>
                    <span className="whitespace-nowrap">+</span>
                    <input
                        type="range"
                        min="0"
                        max="1000"
                        step="50"
                        value={extraAudioDelayMs}
                        onChange={(e) => setExtraAudioDelayMs(Number(e.target.value))}
                        className="w-16 md:w-24 accent-blue-500"
                    />
                    <span className="font-mono text-blue-400 font-semibold text-right whitespace-nowrap">
                        {autoSyncAudio ? `${latency}+${extraAudioDelayMs}=` : ''}{audioDelayMs}ms
                    </span>
                </div>
                <div className="flex items-center gap-2 text-xs md:text-sm text-gray-300">
                    <span className="whitespace-nowrap">Exposure:</span>
                    <input
                        type="range"
                        min="-40"
                        max="40"
                        step="1"
                        value={exposureAdjust}
                        onChange={(e) => setExposureAdjust(Number(e.target.value))}
                        className="w-16 md:w-24 accent-blue-500"
                    />
                    <span className="font-mono text-blue-400 font-semibold text-right">
                        {exposureAdjust > 0 ? `+${exposureAdjust}` : exposureAdjust}
                    </span>
                </div>
            </div>

            {/* Session Info */}
            <div className="mb-3 md:mb-4 grid grid-cols-1 md:grid-cols-2 gap-2 md:gap-4 text-xs md:text-sm">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                    <div className="text-gray-400">Session ID</div>
                    <div className="text-white font-mono break-all">...{sessionId.slice(-6)}</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                    <div className="text-gray-400">MJPEG URL (Virtual Camera)</div>
                    <div className="text-white font-mono break-all">
                        {serverUrl ? getMaskedMjpegUrl(`${serverUrl}/mjpeg/${sessionId}`) : '...'}
                    </div>
                </div>
            </div>

            {/* Video Display */}
            <div className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 md:gap-4">
                {/* Original Feed */}
                <div
                    className={`bg-gray-800 rounded-lg overflow-hidden border border-gray-700 transition-all duration-300 ${fullScreenView === 'original' ? 'fixed inset-0 z-50 !rounded-none m-0' : ''} ${fullScreenView === 'processed' ? 'hidden' : ''}`}
                    onDoubleClick={() => fullScreenView === 'original' && setFullScreenView(null)}
                >
                    <div className="bg-gray-700 px-4 py-2 border-b border-gray-600 flex justifying-between items-center">
                        <h3 className="font-semibold text-white flex-1">Original Feed</h3>
                        <button
                            onClick={() => toggleFullScreen('original')}
                            className="p-1 hover:bg-gray-600 rounded text-gray-300 hover:text-white transition-colors"
                            title={fullScreenView === 'original' ? "Minimize" : "Maximize"}
                        >
                            {fullScreenView === 'original' ? (
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                            ) : (
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>
                            )}
                        </button>
                    </div>
                    <div className={`aspect-video bg-black flex items-center justify-center ${fullScreenView === 'original' ? 'h-[calc(100%-40px)] w-full' : ''}`}>
                        {stream ? (
                            <video
                                ref={originalVideoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full h-full object-contain"
                            />
                        ) : (
                            <div className="text-center text-gray-500">
                                <svg className="w-10 h-10 md:w-16 md:h-16 mx-auto mb-2 md:mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                <p className="text-sm">Camera not active</p>
                            </div>
                        )}
                    </div>
                </div>

                {/* Processed Feed */}
                <div
                    className={`bg-gray-800 rounded-lg overflow-hidden border border-gray-700 transition-all duration-300 ${fullScreenView === 'processed' ? 'fixed inset-0 z-50 !rounded-none m-0' : ''} ${fullScreenView === 'original' ? 'hidden' : ''}`}
                    onDoubleClick={() => fullScreenView === 'processed' && setFullScreenView(null)}
                >
                    <div className="bg-gray-700 px-4 py-2 border-b border-gray-600 flex justify-between items-center">
                        <h3 className="font-semibold text-white flex-1">AI Preview (Post-Surgery)</h3>
                        <div className="flex items-center gap-1">
                            {/* Popout window — regular resizable window user can place anywhere */}
                            {isStreaming && (
                                <button
                                    onClick={handlePopOut}
                                    className={`px-2 py-1 text-xs rounded font-medium transition-colors flex items-center gap-1 ${
                                        isPoppedOut
                                            ? 'bg-blue-600 text-white hover:bg-blue-700'
                                            : 'bg-gray-600 text-gray-200 hover:bg-gray-500'
                                    }`}
                                    title={isPoppedOut ? 'Close pop-out window' : 'Open in separate window — drag it anywhere, use as OBS Window Source'}
                                >
                                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                    </svg>
                                    {isPoppedOut ? 'Close Window' : 'Open Window'}
                                </button>
                            )}
                            <button
                                onClick={() => toggleFullScreen('processed')}
                                className="p-1 hover:bg-gray-600 rounded text-gray-300 hover:text-white transition-colors"
                                title={fullScreenView === 'processed' ? "Minimize" : "Maximize"}
                            >
                                {fullScreenView === 'processed' ? (
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                                ) : (
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" /></svg>
                                )}
                            </button>
                        </div>
                    </div>
                    <div className={`aspect-video bg-black flex items-center justify-center ${fullScreenView === 'processed' ? 'h-[calc(100%-40px)] w-full' : ''}`}>
                        {isStreaming ? (
                            isWsConnected ? (
                                <canvas
                                    ref={wsCanvasRef}
                                    className="w-full h-full object-contain"
                                    style={{ imageRendering: 'auto', filter: processedFrameFilter }}
                                />
                            ) : (
                                <video
                                    ref={(el) => {
                                        processedVideoRef.current = el;
                                        // Start muted ONLY on first mount.
                                        // Don't reset muted on re-renders — the BlackHole
                                        // routing effect sets muted=false after setSinkId.
                                    }}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full h-full object-contain"
                                    style={{ filter: processedFrameFilter }}
                                />
                            )
                        ) : (
                            <div className="text-center text-gray-500">
                                <svg className="w-10 h-10 md:w-16 md:h-16 mx-auto mb-2 md:mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                                <p className="text-sm">AI processing inactive</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Errors */}
            {(webcamError || rtcError) && (
                <div className="mt-3 md:mt-4 p-3 md:p-4 bg-red-900/50 border border-red-700 rounded-lg">
                    <p className="text-red-200 text-sm">
                        {webcamError || rtcError}
                    </p>
                </div>
            )}



            {/* Diagnostics */}
            <div className="mt-3 md:mt-4 p-3 md:p-4 bg-gray-800 border border-gray-700 rounded-lg text-[10px] md:text-xs text-gray-300 space-y-1">
                <div><span className="text-gray-400">Health:</span> {diagnostics.health || '—'}</div>
                <div><span className="text-gray-400">Upload:</span> {diagnostics.upload || '—'}</div>
                <div><span className="text-gray-400">Settings:</span> {diagnostics.settings || '—'}</div>
                <div><span className="text-gray-400">WebRTC:</span> {diagnostics.webrtc || '—'}</div>
                <div>
                    <span className="text-gray-400">Local media:</span>{' '}
                    {diagnostics.localMedia
                        ? `video=${diagnostics.localMedia.videoTracks}, audio=${diagnostics.localMedia.audioTracks}${diagnostics.localMedia.resolution ? ` (${diagnostics.localMedia.resolution})` : ''}`
                        : '—'}
                </div>
                <div>
                    <span className="text-gray-400">Remote media:</span>{' '}
                    {typeof diagnostics.remoteMedia === 'string'
                        ? diagnostics.remoteMedia
                        : diagnostics.remoteMedia
                            ? `video=${diagnostics.remoteMedia.videoTracks}, audio=${diagnostics.remoteMedia.audioTracks}`
                            : '—'}
                </div>
            </div>
        </div>
    );
}

export default CameraView;
