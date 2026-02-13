import React, { useRef, useEffect, useState, useCallback } from 'react';
import useWebcam from '../hooks/useWebcam';
import useWebRTC from '../hooks/useWebRTC';
import useWebSocket from '../hooks/useWebSocket';

function CameraView({ serverUrl, targetImage, isStreaming, setIsStreaming }) {
    const originalVideoRef = useRef(null);
    const processedVideoRef = useRef(null);
    const wsImgRef = useRef(null);
    const wsFrameCountRef = useRef(0);
    const wsLastFpsTimeRef = useRef(performance.now());
    const pendingFrameRef = useRef(false); // Back-pressure: true while waiting for server response

    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [lipSyncEnabled, setLipSyncEnabled] = useState(true);
    const [diagnostics, setDiagnostics] = useState({
        health: null,
        upload: null,
        settings: null,
        webrtc: null,
        localMedia: null,
        remoteMedia: null
    });

    // Custom hooks for webcam and WebSocket
    const { stream, error: webcamError, startWebcam, stopWebcam } = useWebcam(true);
    // WebSocket hook – render into the dedicated <img> ref
    const handleWsFrame = useCallback((base64Frame, wsLatency) => {
        // Clear back-pressure flag — server has responded, we can send next frame
        pendingFrameRef.current = false;
        if (wsImgRef.current) {
            wsImgRef.current.src = `data:image/jpeg;base64,${base64Frame}`;
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

    // Send frames via WebSocket if connected – throttled to ~24 FPS
    useEffect(() => {
        if (!isStreaming || !isWsConnected || !originalVideoRef.current) return;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const video = originalVideoRef.current;

        // Adaptive frame sending: don't send a new frame until the server
        // has responded to the previous one. This prevents queue buildup
        // and naturally adapts to network + GPU processing speed.
        let active = true;
        const MAX_WIDTH = 480;       // 480px is enough for face detection + swap
        const JPEG_QUALITY = 0.5;    // Lower = smaller payload = faster network
        const MIN_INTERVAL = 30;     // Cap at ~33 FPS even if server is super fast

        const sendLoop = () => {
            if (!active) return;
            if (pendingFrameRef.current) {
                // Server hasn't responded yet — skip this frame
                setTimeout(sendLoop, MIN_INTERVAL);
                return;
            }
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
                canvas.width = video.videoWidth * scale;
                canvas.height = video.videoHeight * scale;

                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

                const base64 = canvas.toDataURL('image/jpeg', JPEG_QUALITY).split(',')[1];
                pendingFrameRef.current = true;
                sendWsFrame(base64);
            }
            setTimeout(sendLoop, MIN_INTERVAL);
        };
        sendLoop();

        return () => { active = false; pendingFrameRef.current = false; };
    }, [isStreaming, isWsConnected, sendWsFrame]);

    // Auto re-upload target image when the user switches images mid-stream
    useEffect(() => {
        if (!isStreaming || !serverUrl || !targetImage) return;

        const reupload = async () => {
            try {
                setDiagnostics(prev => ({ ...prev, upload: 'Re-uploading target image...' }));
                const formData = new FormData();
                formData.append('file', targetImage.file);
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
            } catch (err) {
                setDiagnostics(prev => ({ ...prev, upload: `Re-upload error: ${err.message}` }));
            }
        };
        reupload();
    }, [targetImage, isStreaming, serverUrl, sessionId]);

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
            setDiagnostics(prev => ({ ...prev, upload: 'Uploading target image...' }));
            // Upload target image to server
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

            // Apply session settings
            setDiagnostics(prev => ({ ...prev, settings: 'Applying session settings...' }));
            const settingsResponse = await fetch(`${serverUrl}/session/settings?session_id=${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable_lipsync: lipSyncEnabled })
            });
            if (!settingsResponse.ok) {
                const settingsBody = await settingsResponse.json().catch(() => null);
                const message = settingsBody?.error || 'Failed to apply session settings';
                setDiagnostics(prev => ({
                    ...prev,
                    settings: `Settings failed: ${message} (HTTP ${settingsResponse.status})`
                }));
                throw new Error(message);
            }
            setDiagnostics(prev => ({ ...prev, settings: 'Settings applied' }));

            // Start webcam and WebSocket
            const mediaStream = await startWebcam();
            if (!mediaStream) {
                throw new Error('Unable to access webcam/microphone');
            }
            setDiagnostics(prev => ({ ...prev, webrtc: 'Connecting (offer)...' }));
            await connect(mediaStream);
            setDiagnostics(prev => ({ ...prev, webrtc: 'WebRTC connected' }));
            setIsStreaming(true);
        } catch (error) {
            setDiagnostics(prev => ({ ...prev, webrtc: `WebRTC error: ${error.message}` }));
            alert(`Error: ${error.message}`);
        }
    };

    const handleStop = () => {
        stopWebcam();
        disconnect();
        disconnectWs();
        setIsStreaming(false);
        setFps(0);
        setLatency(0);
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
                        <div className="flex items-center gap-2">
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : isWsConnected ? 'bg-blue-500 animate-pulse' : 'bg-red-500'}`}></span>
                            <span className="text-gray-400">
                                {isConnected ? 'WebRTC' : isWsConnected ? 'WebSocket' : 'Disconnected'}
                                ({isConnected ? connectionState : isWsConnected ? 'connected' : 'failed'})
                            </span>
                        </div>
                    </div>
                )}
            </div>

            {/* Session Info */}
            <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                    <div className="text-gray-400">Session ID</div>
                    <div className="text-white font-mono break-all">{sessionId}</div>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-lg p-3">
                    <div className="text-gray-400">MJPEG URL (Virtual Camera)</div>
                    <div className="text-white font-mono break-all">
                        {serverUrl ? `${serverUrl}/mjpeg/${sessionId}` : 'Set server URL'}
                    </div>
                </div>
            </div>

            {/* Video Display */}
            <div className="flex-1 grid grid-cols-2 gap-4">
                {/* Original Feed */}
                <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
                    <div className="bg-gray-700 px-4 py-2 border-b border-gray-600">
                        <h3 className="font-semibold text-white">Original Feed</h3>
                    </div>
                    <div className="aspect-video bg-black flex items-center justify-center">
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
                <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
                    <div className="bg-gray-700 px-4 py-2 border-b border-gray-600">
                        <h3 className="font-semibold text-white">AI Preview (Post-Surgery)</h3>
                    </div>
                    <div className="aspect-video bg-black flex items-center justify-center">
                        {isStreaming ? (
                            isWsConnected ? (
                                <img
                                    ref={wsImgRef}
                                    className="w-full h-full object-contain"
                                    alt="Live Feed"
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
                    {diagnostics.remoteMedia
                        ? `video=${diagnostics.remoteMedia.videoTracks}, audio=${diagnostics.remoteMedia.audioTracks}`
                        : '—'}
                </div>
            </div>
        </div>
    );
}

export default CameraView;
