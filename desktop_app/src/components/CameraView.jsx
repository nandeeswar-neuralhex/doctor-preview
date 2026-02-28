import React, { useRef, useEffect, useState, useCallback } from 'react';
import useWebcam from '../hooks/useWebcam';
import useWebRTC from '../hooks/useWebRTC';
import useWebSocket from '../hooks/useWebSocket';
import { getHardwareProfile, getCachedProfile } from '../hooks/useHardwareProfile';
import { FrameEncoder } from '../hooks/FrameEncoder';

function CameraView({ serverUrl, targetImage, allTargetImages, isStreaming, setIsStreaming }) {
    const originalVideoRef = useRef(null);
    const processedVideoRef = useRef(null);
    const wsCanvasRef = useRef(null);  // Direct canvas paint — zero flicker
    const wsFrameCountRef = useRef(0);
    const wsLastFpsTimeRef = useRef(performance.now());
    const audioCtxRef = useRef(null);
    const audioBufferRef = useRef(new Int16Array(0));
    const audioSampleRateRef = useRef(16000);
    const frameEncoderRef = useRef(null); // Off-main-thread JPEG encoder

    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [lipSyncEnabled, setLipSyncEnabled] = useState(true);
    const [audioDelayMs, setAudioDelayMs] = useState(300);
    const [hwProfile, setHwProfile] = useState(null); // Hardware detection result
    const [diagnostics, setDiagnostics] = useState({
        health: null,
        upload: null,
        settings: null,
        webrtc: null,
        localMedia: null,
        remoteMedia: null,
        hardware: null
    });
    const [fullScreenView, setFullScreenView] = useState(null); // 'original' | 'processed' | null

    // ── Run hardware detection on mount ──
    useEffect(() => {
        getHardwareProfile().then(profile => {
            setHwProfile(profile);
            setDiagnostics(prev => ({
                ...prev,
                hardware: `${profile.tier.toUpperCase()} — ${profile.cores} cores, ${profile.memory}GB RAM, encode=${profile.encodeMsAvg}ms`
            }));
        });
        // Initialize frame encoder
        frameEncoderRef.current = new FrameEncoder();
        return () => {
            if (frameEncoderRef.current) {
                frameEncoderRef.current.dispose();
                frameEncoderRef.current = null;
            }
        };
    }, []);

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

    // Custom hooks for webcam and WebSocket — adaptive capture resolution
    const captureW = hwProfile?.capture?.width || 1280;
    const captureH = hwProfile?.capture?.height || 720;
    const { stream, error: webcamError, startWebcam, stopWebcam } = useWebcam(true, audioDelayMs, captureW, captureH);
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
                    }
                    ctx.drawImage(bitmap, 0, 0);
                    bitmap.close();
                    if (diagnostics.remoteMedia === null) {
                        setDiagnostics(prev => ({ ...prev, remoteMedia: 'Receiving frames' }));
                    }
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
        error: wsError,
        networkQuality,
        configureFromProfile,
        adaptSendRate,
        getAdaptiveParams,
    } = useWebSocket(serverUrl, sessionId, handleWsFrame);

    // Configure WebSocket adaptive sender when hardware profile is ready
    useEffect(() => {
        if (hwProfile) {
            configureFromProfile(hwProfile);
        }
    }, [hwProfile, configureFromProfile]);

    const {
        isConnected,
        error: rtcError,
        connectionState,
        connect,
        disconnect
    } = useWebRTC(serverUrl, sessionId, (remoteStream) => {
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
    });

    // Fallback logic: If WebRTC fails or disconnects, try WebSocket
    useEffect(() => {
        if (isStreaming && !isConnected && connectionState === 'failed' && !isWsConnected) {
            console.log('WebRTC failed, falling back to WebSocket...');
            setDiagnostics(prev => ({ ...prev, webrtc: 'WebRTC failed, trying WebSocket...' }));
            connectWs();
        }
    }, [isStreaming, isConnected, connectionState, isWsConnected, connectWs]);

    // ── ADAPTIVE frame send loop (Google-Meet-grade) ──
    // Uses FrameEncoder (off-main-thread JPEG) + adaptive FPS/quality from backpressure
    useEffect(() => {
        if (!isStreaming || !isWsConnected || !originalVideoRef.current) return;

        const video = originalVideoRef.current;
        const encoder = frameEncoderRef.current;
        let active = true;
        let encoding = false; // Prevent overlapping encodes

        const sendLoop = async () => {
            if (!active) return;

            // Run adaptive backpressure engine — adjusts FPS + quality each tick
            adaptSendRate();
            const params = getAdaptiveParams();
            const interval = 1000 / params.currentFps;

            if (video.readyState === video.HAVE_ENOUGH_DATA && !encoding) {
                encoding = true;
                try {
                    // Compute send dimensions: scale video down to profile width
                    const maxW = hwProfile?.capture?.width || 854;
                    const scale = Math.min(1, maxW / video.videoWidth);
                    const w = Math.round(video.videoWidth * scale);
                    const h = Math.round(video.videoHeight * scale);

                    // Encode JPEG off-main-thread via FrameEncoder
                    const blob = encoder
                        ? await encoder.encode(video, w, h, params.currentQuality)
                        : await new Promise(resolve => {
                            // Ultimate fallback: inline canvas.toBlob
                            const c = document.createElement('canvas');
                            c.width = w; c.height = h;
                            c.getContext('2d').drawImage(video, 0, 0, w, h);
                            c.toBlob(resolve, 'image/jpeg', params.currentQuality);
                        });

                    if (active && blob) {
                        sendWsFrame(blob, audioBufferRef.current, audioSampleRateRef.current);
                    }
                } catch (err) {
                    // Don't break the loop on a single encode error
                    console.warn('[SendLoop] Encode error:', err.message);
                } finally {
                    encoding = false;
                }
            }

            if (active) setTimeout(sendLoop, interval);
        };
        sendLoop();

        return () => { active = false; };
    }, [isStreaming, isWsConnected, sendWsFrame, adaptSendRate, getAdaptiveParams, hwProfile]);

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

        return () => {
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
        if (!isStreaming || !processedVideoRef.current) return;

        const video = processedVideoRef.current;
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
    }, [isStreaming]);

    // Measure round-trip latency via WebRTC stats
    useEffect(() => {
        if (!isStreaming || !isConnected) {
            setLatency(0);
            return;
        }

        // Poll RTCPeerConnection stats every 2 seconds
        const interval = setInterval(async () => {
            try {
                // Access peer connection from the hook's internal ref isn't possible,
                // so we estimate latency from frame timestamps
                const video = processedVideoRef.current;
                if (video && video.getVideoPlaybackQuality) {
                    const quality = video.getVideoPlaybackQuality();
                    // Use totalVideoFrames vs droppedVideoFrames as a proxy
                    const dropped = quality.droppedVideoFrames || 0;
                    const total = quality.totalVideoFrames || 1;
                    const dropRate = (dropped / total) * 100;
                    // If drop rate > 10%, latency is likely high
                    if (dropRate > 10) setLatency(prev => Math.min(prev + 5, 500));
                    else setLatency(prev => Math.max(prev - 5, 0));
                }
            } catch (_) { /* ignore */ }
        }, 2000);

        return () => clearInterval(interval);
    }, [isStreaming, isConnected]);

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

            // Start webcam
            const mediaStream = await startWebcam();
            if (!mediaStream) {
                throw new Error('Unable to access webcam/microphone');
            }

            // Connect via WebSocket directly (runpod_v2 doesn't support WebRTC)
            setDiagnostics(prev => ({ ...prev, webrtc: 'Connecting via WebSocket...' }));
            connectWs();
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
        setDiagnostics(prev => ({
            ...prev,
            health: null, upload: null, settings: null, webrtc: null,
            localMedia: null, remoteMedia: null
            // Keep hardware diagnostics — they don't change per session
        }));
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
            <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-4">
                    {!isStreaming ? (
                        <button
                            onClick={handleStart}
                            disabled={!serverUrl || !targetImage}
                            className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path d="M6.3 2.841A1.5 1.5 0 004 4.11V15.89a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z" />
                            </svg>
                            Start Preview
                        </button>
                    ) : (
                        <button
                            onClick={handleStop}
                            className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition-colors flex items-center gap-2"
                        >
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
                            </svg>
                            Stop Preview
                        </button>
                    )}
                </div>

                {/* Settings quick toggles */}
                <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-300">
                        <input
                            type="checkbox"
                            checked={lipSyncEnabled}
                            onChange={(e) => setLipSyncEnabled(e.target.checked)}
                            disabled={isStreaming}
                        />
                        Lip Sync
                    </label>
                    <div className="flex items-center gap-2 text-sm text-gray-300">
                        <span className="whitespace-nowrap">Audio Delay:</span>
                        <input
                            type="range"
                            min="0"
                            max="1000"
                            step="50"
                            value={audioDelayMs}
                            onChange={(e) => setAudioDelayMs(Number(e.target.value))}
                            className="w-24 accent-blue-500"
                        />
                        <span className="font-mono text-blue-400 font-semibold w-14 text-right">{audioDelayMs}ms</span>
                    </div>
                    <button
                        onClick={handleHealthCheck}
                        className="px-3 py-2 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                    >
                        Check /health
                    </button>
                </div>

                {/* Stats */}
                {isStreaming && (
                    <div className="flex gap-6 text-sm">
                        <div className="flex items-center gap-2">
                            <span className="text-gray-400">FPS:</span>
                            <span className="font-mono text-green-400 font-semibold">{fps}</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="text-gray-400">Latency:</span>
                            <span className="font-mono text-blue-400 font-semibold">{latency}ms</span>
                        </div>
                        {/* Network quality indicator — like Google Meet's signal bars */}
                        <div className="flex items-center gap-1.5" title={`Network: ${networkQuality}`}>
                            {['poor', 'fair', 'good', 'excellent'].map((level, i) => {
                                const active = ['poor', 'fair', 'good', 'excellent'].indexOf(networkQuality) >= i;
                                const colors = ['bg-red-500', 'bg-yellow-500', 'bg-green-400', 'bg-green-400'];
                                return (
                                    <div
                                        key={level}
                                        className={`rounded-sm ${active ? colors[i] : 'bg-gray-600'}`}
                                        style={{ width: 4, height: 6 + i * 4 }}
                                    />
                                );
                            })}
                            <span className="text-gray-400 text-xs ml-1">
                                {networkQuality === 'excellent' ? '' : networkQuality === 'good' ? '' : networkQuality}
                            </span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : isWsConnected ? 'bg-blue-500 animate-pulse' : 'bg-red-500'}`}></span>
                            <span className="text-gray-400">
                                {isConnected ? 'WebRTC' : isWsConnected ? 'WebSocket' : 'Disconnected'}
                            </span>
                        </div>
                        {hwProfile && (
                            <div className="flex items-center gap-1">
                                <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${
                                    hwProfile.tier === 'high' ? 'bg-green-900 text-green-300' :
                                    hwProfile.tier === 'medium' ? 'bg-yellow-900 text-yellow-300' :
                                    'bg-red-900 text-red-300'
                                }`}>
                                    {hwProfile.tier.toUpperCase()}
                                </span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Session Info */}
            <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
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
            <div className="flex-1 grid grid-cols-2 gap-4">
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
                                <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                </svg>
                                <p>Camera not active</p>
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
                    <div className={`aspect-video bg-black flex items-center justify-center ${fullScreenView === 'processed' ? 'h-[calc(100%-40px)] w-full' : ''}`}>
                        {isStreaming ? (
                            isWsConnected ? (
                                <canvas
                                    ref={wsCanvasRef}
                                    className="w-full h-full object-contain"
                                    style={{ imageRendering: 'auto' }}
                                />
                            ) : (
                                <video
                                    ref={processedVideoRef}
                                    autoPlay
                                    playsInline
                                    muted
                                    className="w-full h-full object-contain"
                                />
                            )
                        ) : (
                            <div className="text-center text-gray-500">
                                <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                                </svg>
                                <p>AI processing inactive</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* Errors */}
            {(webcamError || rtcError) && (
                <div className="mt-4 p-4 bg-red-900/50 border border-red-700 rounded-lg">
                    <p className="text-red-200">
                        {webcamError || rtcError}
                    </p>
                </div>
            )}

            {/* Diagnostics */}
            <div className="mt-4 p-4 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 space-y-1">
                <div><span className="text-gray-400">Hardware:</span> {diagnostics.hardware || '—'}</div>
                <div><span className="text-gray-400">Health:</span> {diagnostics.health || '—'}</div>
                <div><span className="text-gray-400">Upload:</span> {diagnostics.upload || '—'}</div>
                <div><span className="text-gray-400">Settings:</span> {diagnostics.settings || '—'}</div>
                <div><span className="text-gray-400">WebRTC:</span> {diagnostics.webrtc || '—'}</div>
                <div>
                    <span className="text-gray-400">Local media:</span>{' '}
                    {diagnostics.localMedia
                        ? `video=${diagnostics.localMedia.videoTracks}, audio=${diagnostics.localMedia.audioTracks}`
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
