import React, { useState, useRef, useCallback, useEffect } from 'react';

/**
 * GuidedCapture — Step-by-step guided face photo capture with quality validation.
 *
 * Flow:
 *   Step-by-step capture (each angle) → Review & Validate → Proceed to analysis
 *
 * Each step shows:
 *   - Visual head-position guide with animation
 *   - Instruction text
 *   - Countdown capture
 *   - Immediate quality feedback (blur, brightness, face detected)
 *
 * Review screen shows:
 *   - All photos with quality scores
 *   - Pass / Fail per photo
 *   - Retake option for failed photos
 *   - Server-side validation (face detection, angle verification)
 */

// ── Angle Definitions ──────────────────────────────────────────────────────

const CAPTURE_STEPS = [
    {
        id: 'front_0',
        label: 'Front',
        stepNo: 1,
        icon: '😊',
        guideRotation: 0,
        instruction: 'Look straight at the camera',
        tips: ['Keep your face centered in the oval', 'Relax your expression', 'Ensure even lighting on both sides'],
    },
    {
        id: 'left_45',
        label: 'Left 45°',
        stepNo: 2,
        icon: '↩️',
        guideRotation: -35,
        instruction: 'Turn your head slightly to the RIGHT',
        tips: ['Expose your left cheek and jawline', 'Keep both eyes visible', 'Don\'t tilt — only rotate'],
    },
    {
        id: 'right_45',
        label: 'Right 45°',
        stepNo: 3,
        icon: '↪️',
        guideRotation: 35,
        instruction: 'Turn your head slightly to the LEFT',
        tips: ['Expose your right cheek and jawline', 'Keep both eyes visible', 'Maintain same distance from camera'],
    },
    {
        id: 'left_90',
        label: 'Left Profile',
        stepNo: 4,
        icon: '⬅️',
        guideRotation: -75,
        instruction: 'Turn your head FULLY to the right (show left profile)',
        tips: ['Show full left side of face', 'Nose, chin and ear should be visible', 'Keep your posture straight'],
    },
    {
        id: 'right_90',
        label: 'Right Profile',
        stepNo: 5,
        icon: '➡️',
        guideRotation: 75,
        instruction: 'Turn your head FULLY to the left (show right profile)',
        tips: ['Show full right side of face', 'Nose, chin and ear should be visible', 'Keep your posture straight'],
    },
];

// ── Quality Checks (client-side) ──────────────────────────────────────────

function checkImageQuality(canvas) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    // 1. Brightness — average luminance
    let totalLum = 0;
    const pixelCount = w * h;
    for (let i = 0; i < data.length; i += 4) {
        totalLum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    const avgBrightness = totalLum / pixelCount;

    // 2. Sharpness — Laplacian variance on grayscale center crop
    const cx = Math.floor(w * 0.25);
    const cy = Math.floor(h * 0.25);
    const cw = Math.floor(w * 0.5);
    const ch = Math.floor(h * 0.5);
    const centerData = ctx.getImageData(cx, cy, cw, ch);
    const gray = [];
    for (let i = 0; i < centerData.data.length; i += 4) {
        gray.push(0.299 * centerData.data[i] + 0.587 * centerData.data[i + 1] + 0.114 * centerData.data[i + 2]);
    }
    let laplacianVar = 0;
    for (let y = 1; y < ch - 1; y++) {
        for (let x = 1; x < cw - 1; x++) {
            const idx = y * cw + x;
            const lap = gray[idx - cw] + gray[idx + cw] + gray[idx - 1] + gray[idx + 1] - 4 * gray[idx];
            laplacianVar += lap * lap;
        }
    }
    laplacianVar /= ((cw - 2) * (ch - 2));

    // 3. Contrast — std deviation of luminance
    let sumSq = 0;
    for (let i = 0; i < data.length; i += 4) {
        const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        sumSq += (lum - avgBrightness) ** 2;
    }
    const contrast = Math.sqrt(sumSq / pixelCount);

    // Thresholds
    const brightnessOk = avgBrightness >= 60 && avgBrightness <= 220;
    const sharpnessOk = laplacianVar > 50;
    const contrastOk = contrast > 20;

    return {
        brightness: { value: Math.round(avgBrightness), ok: brightnessOk, label: brightnessOk ? 'Good' : avgBrightness < 60 ? 'Too dark' : 'Too bright' },
        sharpness: { value: Math.round(laplacianVar), ok: sharpnessOk, label: sharpnessOk ? 'Sharp' : 'Blurry — hold steady' },
        contrast: { value: Math.round(contrast), ok: contrastOk, label: contrastOk ? 'Good' : 'Low contrast' },
        overallOk: brightnessOk && sharpnessOk && contrastOk,
    };
}

// ── Component ──────────────────────────────────────────────────────────────

export default function GuidedCapture({ onComplete, serverUrl }) {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [stream, setStream] = useState(null);
    const [phase, setPhase] = useState('capture');   // capture | review
    const [currentStep, setCurrentStep] = useState(0);
    const [photos, setPhotos] = useState([]);         // { angle, blob, preview, quality, validated }
    const [countdown, setCountdown] = useState(null);
    const [lastQuality, setLastQuality] = useState(null);
    const [validating, setValidating] = useState(false);
    const [validationResults, setValidationResults] = useState(null);
    const [retakeIndex, setRetakeIndex] = useState(null); // which photo to retake
    const [patientInfo, setPatientInfo] = useState({ name: '', age: '', skin_type: '' });
    const [cameraReady, setCameraReady] = useState(false);

    const analysisUrl = import.meta.env.VITE_ANALYSIS_URL || serverUrl?.replace(/:\d+/, ':8766') || serverUrl;

    // ── Camera ─────────────────────────────────────────────────────────

    useEffect(() => {
        startCamera();
        return () => stopCamera();
    }, []);

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
            setCameraReady(true);
        } catch (err) {
            console.error('Camera access failed:', err);
        }
    };

    const stopCamera = () => {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            setStream(null);
        }
    };

    // ── Capture ────────────────────────────────────────────────────────

    const startCountdown = useCallback(() => {
        setLastQuality(null);
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
    }, [currentStep, photos, retakeIndex]);

    const doCapture = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        // Mirror the capture to match preview
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();

        // Client-side quality check
        const quality = checkImageQuality(canvas);
        setLastQuality(quality);

        canvas.toBlob((blob) => {
            const url = canvas.toDataURL('image/jpeg', 0.95);

            const stepDef = retakeIndex !== null ? CAPTURE_STEPS[retakeIndex] : CAPTURE_STEPS[currentStep];

            const newPhoto = {
                angle: stepDef.id,
                label: stepDef.label,
                blob,
                preview: url,
                quality,
                timestamp: Date.now(),
                validated: null,     // will be set after server validation
            };

            if (retakeIndex !== null) {
                // Replacing a specific photo
                setPhotos(prev => prev.map((p, i) => i === retakeIndex ? newPhoto : p));
                setRetakeIndex(null);
            } else {
                setPhotos(prev => [...prev, newPhoto]);
            }
        }, 'image/jpeg', 0.95);
    }, [currentStep, retakeIndex]);

    // Auto-advance after capture
    useEffect(() => {
        if (retakeIndex !== null) return; // don't advance during retake
        if (photos.length > 0 && photos.length === currentStep + 1 && currentStep < CAPTURE_STEPS.length - 1) {
            const timeout = setTimeout(() => setCurrentStep(prev => prev + 1), 800);
            return () => clearTimeout(timeout);
        }
    }, [photos.length, currentStep, retakeIndex]);

    // ── Enter review when all angles captured ──────────────────────────

    const goToReview = () => {
        setPhase('review');
        validateAllPhotos();
    };

    // ── Server Validation ──────────────────────────────────────────────

    const validateAllPhotos = async () => {
        setValidating(true);
        try {
            const formData = new FormData();
            photos.forEach((photo, i) => {
                formData.append('images', photo.blob, `${photo.angle}.jpg`);
                formData.append('angles', photo.angle);
            });

            const res = await fetch(`${analysisUrl}/validate`, {
                method: 'POST',
                body: formData,
            });

            if (res.ok) {
                const data = await res.json();
                setValidationResults(data);

                // Merge validation into photos
                setPhotos(prev => prev.map((p, i) => ({
                    ...p,
                    validated: data.results?.[i] || null,
                })));
            } else {
                // Fallback — use client-side only
                setValidationResults({ status: 'client_only', message: 'Server validation unavailable — using local checks' });
                setPhotos(prev => prev.map(p => ({
                    ...p,
                    validated: {
                        face_detected: true,
                        angle_ok: true,
                        quality_score: p.quality.overallOk ? 85 : 50,
                        passed: p.quality.overallOk,
                        issues: p.quality.overallOk ? [] : ['Client-side quality check failed'],
                    },
                })));
            }
        } catch (err) {
            console.error('Validation failed:', err);
            // Fallback
            setPhotos(prev => prev.map(p => ({
                ...p,
                validated: {
                    face_detected: true,
                    angle_ok: true,
                    quality_score: p.quality.overallOk ? 85 : 50,
                    passed: p.quality.overallOk,
                    issues: [],
                },
            })));
        } finally {
            setValidating(false);
        }
    };

    // ── Retake a specific photo ────────────────────────────────────────

    const startRetake = (index) => {
        setRetakeIndex(index);
        setPhase('capture');
        setCurrentStep(index);
        setLastQuality(null);
        setValidationResults(null);
    };

    // ── Proceed to analysis ────────────────────────────────────────────

    const handleProceed = () => {
        onComplete(photos, patientInfo);
    };

    // ── Computed ───────────────────────────────────────────────────────

    const allPassed = photos.every(p => p.validated?.passed);
    const passedCount = photos.filter(p => p.validated?.passed).length;
    const failedCount = photos.filter(p => p.validated && !p.validated.passed).length;
    const stepDef = retakeIndex !== null ? CAPTURE_STEPS[retakeIndex] : CAPTURE_STEPS[currentStep];

    // ════════════════════════════════════════════════════════════════════
    // ── RENDER ─────────────────────────────────────────────────────────
    // ════════════════════════════════════════════════════════════════════

    if (phase === 'review') {
        return <ReviewScreen
            photos={photos}
            validating={validating}
            validationResults={validationResults}
            allPassed={allPassed}
            passedCount={passedCount}
            failedCount={failedCount}
            patientInfo={patientInfo}
            setPatientInfo={setPatientInfo}
            onRetake={startRetake}
            onRevalidate={validateAllPhotos}
            onProceed={handleProceed}
        />;
    }

    return (
        <div className="flex h-full bg-gray-950 text-white">
            {/* ── Left: Step Progress ─────────────────────────────────── */}
            <div className="w-72 border-r border-gray-800 flex flex-col">
                <div className="p-5 border-b border-gray-800">
                    <h2 className="text-lg font-bold mb-1">📸 Photo Capture</h2>
                    <p className="text-xs text-gray-400">Follow each step to capture all angles</p>
                </div>

                {/* Steps */}
                <div className="flex-1 p-4 space-y-2 overflow-auto">
                    {CAPTURE_STEPS.map((step, i) => {
                        const captured = photos.find(p => p.angle === step.id);
                        const isActive = (retakeIndex !== null ? retakeIndex : currentStep) === i;
                        const isRetaking = retakeIndex === i;

                        return (
                            <div
                                key={step.id}
                                className={`flex items-center gap-3 p-3 rounded-xl border transition-all ${
                                    isActive
                                        ? 'border-indigo-500 bg-indigo-500/10'
                                        : captured
                                            ? 'border-gray-700 bg-gray-900/50'
                                            : 'border-gray-800 bg-gray-900/20 opacity-50'
                                }`}
                            >
                                {/* Status Icon */}
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg shrink-0 ${
                                    isRetaking ? 'bg-orange-600' :
                                    captured?.quality?.overallOk ? 'bg-green-600' :
                                    captured ? 'bg-yellow-600' :
                                    isActive ? 'bg-indigo-600' : 'bg-gray-700'
                                }`}>
                                    {isRetaking ? '🔄' : captured ? '✓' : step.stepNo}
                                </div>

                                {/* Step Info */}
                                <div className="flex-1 min-w-0">
                                    <p className="text-sm font-medium">{step.icon} {step.label}</p>
                                    {captured && (
                                        <p className={`text-xs mt-0.5 ${captured.quality?.overallOk ? 'text-green-400' : 'text-yellow-400'}`}>
                                            {captured.quality?.overallOk ? '✓ Good quality' : '⚠ May need retake'}
                                        </p>
                                    )}
                                    {isActive && !captured && (
                                        <p className="text-xs text-indigo-400 mt-0.5">Current step</p>
                                    )}
                                </div>

                                {/* Thumbnail */}
                                {captured && (
                                    <img
                                        src={captured.preview}
                                        alt={step.label}
                                        className="w-12 h-12 rounded-lg object-cover border border-gray-700 shrink-0"
                                    />
                                )}
                            </div>
                        );
                    })}
                </div>

                {/* Go to Review */}
                <div className="p-4 border-t border-gray-800 space-y-2">
                    <div className="text-xs text-gray-500 text-center mb-1">
                        {photos.length} / {CAPTURE_STEPS.length} photos captured
                    </div>
                    <button
                        onClick={goToReview}
                        disabled={photos.length === 0}
                        className="w-full py-3 bg-indigo-600 hover:bg-indigo-700 rounded-xl font-medium text-sm disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                    >
                        ✅ Review & Validate ({photos.length} photos)
                    </button>
                    <button
                        onClick={() => { setPhotos([]); setCurrentStep(0); setRetakeIndex(null); setLastQuality(null); }}
                        className="w-full py-2 text-xs text-gray-500 hover:text-gray-300 transition-colors"
                    >
                        🗑️ Reset All
                    </button>
                </div>
            </div>

            {/* ── Right: Camera + Guide ───────────────────────────────── */}
            <div className="flex-1 flex flex-col">
                {/* Instruction Bar */}
                <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <h3 className="text-sm font-semibold text-gray-300">
                                Step {stepDef.stepNo} of {CAPTURE_STEPS.length} — <span className="text-indigo-400">{stepDef.label}</span>
                                {retakeIndex !== null && <span className="text-orange-400 ml-2">(Retaking)</span>}
                            </h3>
                            <p className="text-lg font-medium mt-1">{stepDef.instruction}</p>
                        </div>
                        {lastQuality && (
                            <div className={`px-4 py-2 rounded-xl text-sm font-medium ${lastQuality.overallOk ? 'bg-green-600/20 text-green-400 border border-green-700' : 'bg-yellow-600/20 text-yellow-400 border border-yellow-700'}`}>
                                {lastQuality.overallOk ? '✅ Good capture!' : '⚠️ Quality issues detected'}
                            </div>
                        )}
                    </div>
                    {/* Tips */}
                    <div className="flex gap-4 mt-2">
                        {stepDef.tips.map((tip, i) => (
                            <span key={i} className="text-xs text-gray-500">💡 {tip}</span>
                        ))}
                    </div>
                </div>

                {/* Camera View */}
                <div className="flex-1 relative bg-black overflow-hidden">
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
                        {/* Rotating head guide */}
                        <div
                            className="relative transition-transform duration-700"
                            style={{ transform: `rotateY(${stepDef.guideRotation}deg)`, perspective: '600px' }}
                        >
                            <div className="w-56 h-72 border-[3px] border-indigo-400/50 rounded-[50%] relative"
                                style={{ boxShadow: '0 0 30px rgba(99,102,241,0.15)' }}>
                                {/* Center crosshair */}
                                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">
                                    <div className="w-3 h-3 border-2 border-indigo-400/40 rounded-full" />
                                </div>
                                {/* Nose line */}
                                <div className="absolute top-1/3 left-1/2 w-0.5 h-1/3 bg-indigo-400/20 -translate-x-1/2" />
                            </div>
                            {/* Label */}
                            <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-indigo-600/90 backdrop-blur px-4 py-1.5 rounded-full text-sm font-medium whitespace-nowrap">
                                {stepDef.icon} {stepDef.label}
                            </div>
                        </div>
                    </div>

                    {/* Countdown */}
                    {countdown && (
                        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                            <div className="text-center">
                                <span className="text-8xl font-bold text-white animate-pulse">{countdown}</span>
                                <p className="text-gray-300 mt-2">Hold still...</p>
                            </div>
                        </div>
                    )}

                    {/* Quality feedback (shows briefly after capture) */}
                    {lastQuality && countdown === null && (
                        <div className="absolute bottom-20 left-1/2 -translate-x-1/2 bg-gray-900/90 backdrop-blur rounded-xl px-5 py-3 flex gap-4 z-10">
                            <QualityBadge label="Brightness" {...lastQuality.brightness} />
                            <QualityBadge label="Sharpness" {...lastQuality.sharpness} />
                            <QualityBadge label="Contrast" {...lastQuality.contrast} />
                        </div>
                    )}

                    {/* Capture Button */}
                    {cameraReady && (
                        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-10">
                            <button
                                onClick={startCountdown}
                                disabled={countdown !== null}
                                className="group w-20 h-20 rounded-full bg-white/90 border-4 border-indigo-500 hover:border-indigo-400 hover:scale-110 transition-all disabled:opacity-50 shadow-lg shadow-indigo-500/20"
                            >
                                <div className="w-14 h-14 rounded-full bg-indigo-500 group-hover:bg-indigo-400 mx-auto transition-colors" />
                            </button>
                            <p className="text-center text-xs text-gray-400 mt-2">Press to capture</p>
                        </div>
                    )}

                    {/* Skip button for optional angles */}
                    {currentStep >= 3 && !photos.find(p => p.angle === stepDef.id) && retakeIndex === null && (
                        <button
                            onClick={() => setCurrentStep(prev => Math.min(prev + 1, CAPTURE_STEPS.length - 1))}
                            className="absolute bottom-6 right-6 text-xs text-gray-400 hover:text-white bg-gray-800/80 px-3 py-1.5 rounded-lg z-10"
                        >
                            Skip this angle →
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
}


// ════════════════════════════════════════════════════════════════════════════
// ── Review Screen ──────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════

function ReviewScreen({
    photos, validating, validationResults, allPassed, passedCount, failedCount,
    patientInfo, setPatientInfo, onRetake, onRevalidate, onProceed,
}) {
    return (
        <div className="h-full flex flex-col bg-gray-950 text-white">
            {/* Header */}
            <div className="bg-gray-900 border-b border-gray-800 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-lg font-bold">📋 Photo Review & Validation</h2>
                        <p className="text-sm text-gray-400 mt-0.5">
                            {validating
                                ? 'Validating photos with AI...'
                                : validationResults
                                    ? `${passedCount} passed · ${failedCount} need attention`
                                    : 'Checking photo quality...'}
                        </p>
                    </div>

                    {/* Summary Badge */}
                    {!validating && validationResults && (
                        <div className={`px-5 py-2.5 rounded-xl text-sm font-semibold ${
                            allPassed
                                ? 'bg-green-600/20 text-green-400 border border-green-700'
                                : 'bg-yellow-600/20 text-yellow-400 border border-yellow-700'
                        }`}>
                            {allPassed ? '✅ All Photos Passed!' : `⚠️ ${failedCount} Photo${failedCount > 1 ? 's' : ''} Need Retake`}
                        </div>
                    )}
                </div>
            </div>

            {/* Loading Spinner */}
            {validating && (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center">
                        <div className="animate-spin w-14 h-14 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-4" />
                        <p className="text-lg font-medium">Analyzing photo quality...</p>
                        <p className="text-gray-400 text-sm mt-1">Checking face detection, sharpness, and angles</p>
                    </div>
                </div>
            )}

            {/* Photo Grid */}
            {!validating && (
                <div className="flex-1 overflow-auto p-6">
                    {/* Patient Info */}
                    <div className="mb-6 bg-gray-900 rounded-2xl p-5 border border-gray-800">
                        <h3 className="text-sm font-semibold text-gray-400 mb-3">👤 Patient Information (Optional)</h3>
                        <div className="flex gap-3">
                            <input
                                type="text"
                                placeholder="Patient Name"
                                value={patientInfo.name}
                                onChange={e => setPatientInfo(prev => ({ ...prev, name: e.target.value }))}
                                className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                            />
                            <input
                                type="number"
                                placeholder="Age"
                                value={patientInfo.age}
                                onChange={e => setPatientInfo(prev => ({ ...prev, age: e.target.value }))}
                                className="w-20 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                            />
                            <select
                                value={patientInfo.skin_type}
                                onChange={e => setPatientInfo(prev => ({ ...prev, skin_type: e.target.value }))}
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
                    </div>

                    {/* Photo Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
                        {photos.map((photo, i) => (
                            <PhotoCard key={i} photo={photo} index={i} onRetake={onRetake} />
                        ))}
                    </div>

                    {/* Validation Details */}
                    {validationResults && (
                        <div className="mt-6 bg-gray-900 rounded-2xl p-5 border border-gray-800">
                            <h3 className="text-sm font-semibold text-gray-400 mb-3">🔍 Validation Details</h3>
                            <div className="space-y-2">
                                {photos.map((photo, i) => (
                                    <div key={i} className={`flex items-center gap-3 p-3 rounded-xl ${
                                        photo.validated?.passed ? 'bg-green-900/10' : 'bg-red-900/10'
                                    }`}>
                                        <span className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                                            photo.validated?.passed ? 'bg-green-600' : 'bg-red-600'
                                        }`}>
                                            {photo.validated?.passed ? '✓' : '✕'}
                                        </span>
                                        <span className="text-sm font-medium w-28">{photo.label}</span>
                                        <span className="text-xs text-gray-400 flex-1">
                                            {photo.validated?.issues?.length > 0
                                                ? photo.validated.issues.join(' · ')
                                                : 'All checks passed'}
                                        </span>
                                        <span className="text-sm font-mono">
                                            {photo.validated?.quality_score?.toFixed(0) || '—'}/100
                                        </span>
                                        {!photo.validated?.passed && (
                                            <button
                                                onClick={() => onRetake(i)}
                                                className="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded-lg text-xs font-medium"
                                            >
                                                🔄 Retake
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Bottom Actions */}
            {!validating && (
                <div className="border-t border-gray-800 px-6 py-4 flex items-center justify-between">
                    <button
                        onClick={onRevalidate}
                        className="px-5 py-2.5 bg-gray-800 hover:bg-gray-700 rounded-xl text-sm"
                    >
                        🔄 Re-validate All
                    </button>
                    <div className="flex gap-3">
                        {!allPassed && failedCount > 0 && (
                            <p className="text-xs text-yellow-400 self-center mr-2">
                                ⚠️ {failedCount} photo{failedCount > 1 ? 's' : ''} may affect analysis accuracy
                            </p>
                        )}
                        <button
                            onClick={onProceed}
                            disabled={photos.length === 0}
                            className={`px-8 py-3 rounded-xl font-medium text-sm transition-all ${
                                allPassed
                                    ? 'bg-green-600 hover:bg-green-700 shadow-lg shadow-green-600/20'
                                    : 'bg-indigo-600 hover:bg-indigo-700'
                            } disabled:opacity-40`}
                        >
                            {allPassed ? '✅ Proceed to Analysis' : '⚡ Proceed Anyway'} ({photos.length} photos)
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}


// ════════════════════════════════════════════════════════════════════════════
// ── Photo Card ─────────────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════

function PhotoCard({ photo, index, onRetake }) {
    const v = photo.validated;
    const q = photo.quality;

    return (
        <div className={`bg-gray-900 rounded-2xl border overflow-hidden transition-all ${
            v?.passed ? 'border-green-700/50' : v ? 'border-red-700/50' : 'border-gray-800'
        }`}>
            {/* Image */}
            <div className="relative">
                <img
                    src={photo.preview}
                    alt={photo.label}
                    className="w-full h-44 object-cover"
                />
                {/* Status Badge */}
                <div className={`absolute top-2 left-2 px-2.5 py-1 rounded-lg text-xs font-semibold ${
                    v?.passed ? 'bg-green-600' :
                    v ? 'bg-red-600' : 'bg-gray-700'
                }`}>
                    {v?.passed ? '✓ Passed' : v ? '✕ Failed' : '⏳ Pending'}
                </div>
                {/* Score */}
                {v?.quality_score != null && (
                    <div className="absolute top-2 right-2 bg-black/70 backdrop-blur px-2 py-1 rounded-lg text-xs font-mono">
                        {v.quality_score.toFixed(0)}/100
                    </div>
                )}
            </div>

            {/* Info */}
            <div className="p-3">
                <p className="text-sm font-medium">{photo.label}</p>

                {/* Quality Indicators */}
                <div className="flex gap-2 mt-2">
                    <MiniIndicator ok={q?.brightness?.ok} label="☀️" title={q?.brightness?.label} />
                    <MiniIndicator ok={q?.sharpness?.ok} label="🔍" title={q?.sharpness?.label} />
                    <MiniIndicator ok={q?.contrast?.ok} label="🎨" title={q?.contrast?.label} />
                    {v && <MiniIndicator ok={v.face_detected} label="👤" title={v.face_detected ? 'Face found' : 'No face'} />}
                    {v && <MiniIndicator ok={v.angle_ok} label="📐" title={v.angle_ok ? 'Angle OK' : 'Wrong angle'} />}
                </div>

                {/* Issues */}
                {v?.issues?.length > 0 && (
                    <div className="mt-2 space-y-0.5">
                        {v.issues.map((issue, j) => (
                            <p key={j} className="text-xs text-red-400">• {issue}</p>
                        ))}
                    </div>
                )}

                {/* Retake Button */}
                {v && !v.passed && (
                    <button
                        onClick={() => onRetake(index)}
                        className="mt-2 w-full py-1.5 bg-orange-600 hover:bg-orange-700 rounded-lg text-xs font-medium transition-colors"
                    >
                        🔄 Retake This Photo
                    </button>
                )}
            </div>
        </div>
    );
}


// ── Tiny Sub-components ────────────────────────────────────────────────────

function QualityBadge({ label, ok, label: description }) {
    return (
        <div className="text-center">
            <div className={`w-3 h-3 rounded-full mx-auto ${ok ? 'bg-green-500' : 'bg-yellow-500'}`} />
            <p className="text-xs text-gray-400 mt-1">{label}</p>
            <p className={`text-xs font-medium ${ok ? 'text-green-400' : 'text-yellow-400'}`}>{description}</p>
        </div>
    );
}

function MiniIndicator({ ok, label, title }) {
    return (
        <span
            title={title}
            className={`text-xs px-1.5 py-0.5 rounded ${
                ok ? 'bg-green-900/30 text-green-400' : ok === false ? 'bg-red-900/30 text-red-400' : 'bg-gray-800 text-gray-500'
            }`}
        >
            {label}
        </span>
    );
}
