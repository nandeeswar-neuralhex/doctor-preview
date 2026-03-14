import React, { useState, useRef, useCallback, useEffect } from 'react';

/**
 * PhotoCapture — Multi-angle guided face photo capture.
 * Guides the user through 5 angles with face alignment overlay.
 */

const ANGLES = [
    { id: 'front_0', label: 'Front', icon: '😊', instruction: 'Look straight at the camera' },
    { id: 'left_45', label: 'Left 45°', icon: '👈', instruction: 'Turn head slightly to your right' },
    { id: 'right_45', label: 'Right 45°', icon: '👉', instruction: 'Turn head slightly to your left' },
    { id: 'left_90', label: 'Left Profile', icon: '⬅️', instruction: 'Turn head fully to your right' },
    { id: 'right_90', label: 'Right Profile', icon: '➡️', instruction: 'Turn head fully to your left' },
];

export default function PhotoCapture({ onAnalyze, patientInfo, onPatientChange, previousSessions, onLoadSession }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [photos, setPhotos] = useState([]);
    const [currentAngle, setCurrentAngle] = useState(0);
    const [captureMode, setCaptureMode] = useState('webcam'); // webcam | upload
    const [countdown, setCountdown] = useState(null);

    // ── Webcam Setup ───────────────────────────────────────────────────────

    useEffect(() => {
        if (captureMode === 'webcam') {
            startCamera();
        }
        return () => stopCamera();
    }, [captureMode]);

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 1920, height: 1080, facingMode: 'user' },
                audio: false,
            });
            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
            }
            setStream(mediaStream);
        } catch (err) {
            console.error('Camera access failed:', err);
            setCaptureMode('upload');
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            setStream(null);
        }
    };

    // ── Capture ────────────────────────────────────────────────────────────

    const capturePhoto = useCallback(() => {
        setCountdown(3);
        const timer = setInterval(() => {
            setCountdown(prev => {
                if (prev <= 1) {
                    clearInterval(timer);
                    doCapture();
                    return null;
                }
                return prev - 1;
            });
        }, 1000);
    }, [currentAngle, photos]);

    const doCapture = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            const url = canvas.toDataURL('image/jpeg', 0.95);
            const newPhoto = {
                angle: ANGLES[currentAngle].id,
                blob,
                preview: url,
                timestamp: Date.now(),
            };
            setPhotos(prev => [...prev, newPhoto]);
            setCurrentAngle(prev => Math.min(prev + 1, ANGLES.length - 1));
        }, 'image/jpeg', 0.95);
    }, [currentAngle]);

    // ── Upload Mode ────────────────────────────────────────────────────────

    const handleFileUpload = (e) => {
        const files = Array.from(e.target.files);
        files.forEach((file, i) => {
            const reader = new FileReader();
            reader.onload = (ev) => {
                setPhotos(prev => [
                    ...prev,
                    {
                        angle: ANGLES[Math.min(prev.length, ANGLES.length - 1)].id,
                        blob: file,
                        preview: ev.target.result,
                        timestamp: Date.now(),
                    },
                ]);
            };
            reader.readAsDataURL(file);
        });
    };

    const removePhoto = (index) => {
        setPhotos(prev => prev.filter((_, i) => i !== index));
    };

    // ── Render ─────────────────────────────────────────────────────────────

    return (
        <div className="flex h-full">
            {/* Left: Camera / Upload */}
            <div className="flex-1 flex flex-col p-6">
                {/* Patient Info Bar */}
                <div className="flex gap-3 mb-4">
                    <input
                        type="text"
                        placeholder="Patient Name"
                        value={patientInfo.name}
                        onChange={e => onPatientChange({ ...patientInfo, name: e.target.value })}
                        className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                    <input
                        type="number"
                        placeholder="Age"
                        value={patientInfo.age}
                        onChange={e => onPatientChange({ ...patientInfo, age: e.target.value })}
                        className="w-20 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    />
                    <select
                        value={patientInfo.skin_type}
                        onChange={e => onPatientChange({ ...patientInfo, skin_type: e.target.value })}
                        className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                    >
                        <option value="">Skin Type</option>
                        <option value="dry">Dry</option>
                        <option value="oily">Oily</option>
                        <option value="combination">Combination</option>
                        <option value="normal">Normal</option>
                        <option value="sensitive">Sensitive</option>
                    </select>
                </div>

                {/* Mode Toggle */}
                <div className="flex gap-2 mb-4">
                    <button
                        onClick={() => setCaptureMode('webcam')}
                        className={`px-4 py-2 rounded-lg text-sm ${captureMode === 'webcam' ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400'}`}
                    >
                        📹 Webcam
                    </button>
                    <button
                        onClick={() => setCaptureMode('upload')}
                        className={`px-4 py-2 rounded-lg text-sm ${captureMode === 'upload' ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-400'}`}
                    >
                        📁 Upload Photos
                    </button>
                </div>

                {/* Camera View */}
                {captureMode === 'webcam' ? (
                    <div className="relative flex-1 bg-black rounded-xl overflow-hidden">
                        <video
                            ref={videoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full h-full object-cover"
                            style={{ transform: 'scaleX(-1)' }}
                        />
                        <canvas ref={canvasRef} className="hidden" />

                        {/* Face Guide Overlay */}
                        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <div className="w-64 h-80 border-2 border-indigo-400/50 rounded-[50%] relative">
                                <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-indigo-600 px-3 py-1 rounded-full text-xs font-medium">
                                    {ANGLES[currentAngle]?.icon} {ANGLES[currentAngle]?.label}
                                </div>
                            </div>
                        </div>

                        {/* Instruction */}
                        <div className="absolute bottom-4 left-0 right-0 text-center">
                            <p className="text-sm text-gray-300 bg-black/60 inline-block px-4 py-2 rounded-lg">
                                {ANGLES[currentAngle]?.instruction}
                            </p>
                        </div>

                        {/* Countdown */}
                        {countdown && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                                <span className="text-7xl font-bold text-white animate-ping">{countdown}</span>
                            </div>
                        )}

                        {/* Capture Button */}
                        <div className="absolute bottom-16 left-1/2 -translate-x-1/2">
                            <button
                                onClick={capturePhoto}
                                disabled={countdown !== null}
                                className="w-16 h-16 rounded-full bg-white border-4 border-indigo-500 hover:scale-110 transition-transform disabled:opacity-50"
                            />
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex items-center justify-center border-2 border-dashed border-gray-700 rounded-xl">
                        <label className="cursor-pointer text-center">
                            <div className="text-5xl mb-3">📸</div>
                            <p className="text-gray-400 mb-2">Click to upload face photos</p>
                            <p className="text-gray-500 text-sm">Up to 5 photos (front + 4 angles)</p>
                            <input
                                type="file"
                                accept="image/*"
                                multiple
                                onChange={handleFileUpload}
                                className="hidden"
                            />
                        </label>
                    </div>
                )}
            </div>

            {/* Right: Photo Queue + Controls */}
            <div className="w-80 border-l border-gray-800 p-4 flex flex-col">
                <h3 className="text-sm font-semibold text-gray-400 mb-3">
                    Captured Photos ({photos.length}/5)
                </h3>

                {/* Angle Progress */}
                <div className="flex gap-1 mb-4">
                    {ANGLES.map((angle, i) => (
                        <div
                            key={angle.id}
                            className={`flex-1 h-1.5 rounded-full ${
                                i < photos.length ? 'bg-indigo-500' : 'bg-gray-700'
                            }`}
                        />
                    ))}
                </div>

                {/* Photo Thumbnails */}
                <div className="flex-1 overflow-auto space-y-2">
                    {photos.map((photo, i) => (
                        <div key={i} className="relative group">
                            <img
                                src={photo.preview}
                                alt={photo.angle}
                                className="w-full h-40 object-cover rounded-lg border border-gray-700"
                            />
                            <div className="absolute top-2 left-2 bg-black/70 px-2 py-0.5 rounded text-xs">
                                {ANGLES.find(a => a.id === photo.angle)?.label || photo.angle}
                            </div>
                            <button
                                onClick={() => removePhoto(i)}
                                className="absolute top-2 right-2 bg-red-600 w-6 h-6 rounded-full text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                            >
                                ✕
                            </button>
                        </div>
                    ))}

                    {photos.length === 0 && (
                        <div className="text-center text-gray-500 mt-10">
                            <p className="text-3xl mb-2">📷</p>
                            <p className="text-sm">No photos yet</p>
                            <p className="text-xs text-gray-600">Capture or upload to begin</p>
                        </div>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="mt-4 space-y-2">
                    <button
                        onClick={() => onAnalyze(photos)}
                        disabled={photos.length === 0}
                        className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                        🔬 Analyze {photos.length > 1 ? `${photos.length} Photos` : 'Photo'}
                    </button>
                    <button
                        onClick={() => { setPhotos([]); setCurrentAngle(0); }}
                        className="w-full py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-400"
                    >
                        🗑️ Clear All
                    </button>
                </div>

                {/* Previous Sessions */}
                {previousSessions.length > 0 && (
                    <div className="mt-4 border-t border-gray-800 pt-3">
                        <h4 className="text-xs font-semibold text-gray-500 mb-2">Previous Sessions</h4>
                        <div className="space-y-1 max-h-32 overflow-auto">
                            {previousSessions.slice(0, 5).map(s => (
                                <button
                                    key={s.session_id}
                                    onClick={() => onLoadSession(s.session_id)}
                                    className="w-full text-left px-3 py-1.5 bg-gray-800/50 hover:bg-gray-800 rounded text-xs"
                                >
                                    <span className="text-gray-300">{s.patient?.name || 'Unknown'}</span>
                                    <span className="text-gray-500 ml-2">Score: {s.overall_score?.toFixed(0) || '—'}</span>
                                </button>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
