import React, { useRef, useEffect, useState } from 'react';
import useWebcam from '../hooks/useWebcam';
import useWebRTC from '../hooks/useWebRTC';

function CameraView({ serverUrl, targetImage, isStreaming, setIsStreaming }) {
    const originalVideoRef = useRef(null);
    const processedVideoRef = useRef(null);

    const [fps, setFps] = useState(0);
    const [latency, setLatency] = useState(0);
    const [sessionId] = useState(() => `session-${Date.now()}`);
    const [lipSyncEnabled, setLipSyncEnabled] = useState(true);

    // Custom hooks for webcam and WebSocket
    const { stream, error: webcamError, startWebcam, stopWebcam } = useWebcam(true);
    const {
        isConnected,
        error: rtcError,
        connect,
        disconnect
    } = useWebRTC(serverUrl, sessionId, (remoteStream) => {
        if (processedVideoRef.current) {
            processedVideoRef.current.srcObject = remoteStream;
        }
    });

    // Display webcam stream in video element
    useEffect(() => {
        if (stream && originalVideoRef.current) {
            originalVideoRef.current.srcObject = stream;
        }
    }, [stream]);

    // FPS from processed video
    useEffect(() => {
        if (!isStreaming || !processedVideoRef.current) return;

        let frameCount = 0;
        let lastTime = performance.now();
        let rafId;

        const tick = (now) => {
            frameCount++;
            if (now - lastTime >= 1000) {
                setFps(frameCount);
                frameCount = 0;
                lastTime = now;
            }
            rafId = requestAnimationFrame(tick);
        };

        rafId = requestAnimationFrame(tick);
        return () => {
            if (rafId) cancelAnimationFrame(rafId);
        };
    }, [isStreaming]);

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
                throw new Error(errorMessage);
            }

            // Apply session settings
            await fetch(`${serverUrl}/session/settings?session_id=${sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable_lipsync: lipSyncEnabled })
            });

            // Start webcam and WebSocket
            const mediaStream = await startWebcam();
            if (!mediaStream) {
                throw new Error('Unable to access webcam/microphone');
            }
            await connect(mediaStream);
            setIsStreaming(true);
        } catch (error) {
            alert(`Error: ${error.message}`);
        }
    };

    const handleStop = () => {
        stopWebcam();
        disconnect();
        setIsStreaming(false);
        setFps(0);
        setLatency(0);
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
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></span>
                            <span className="text-gray-400">{isConnected ? 'Connected' : 'Disconnected'}</span>
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
                            <video
                                ref={processedVideoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full h-full object-contain"
                            />
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
        </div>
    );
}

export default CameraView;
