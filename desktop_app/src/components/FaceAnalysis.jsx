import React, { useState, useCallback, useEffect, useRef, lazy, Suspense } from 'react';
import GuidedCapture from './GuidedCapture';
import PhotoCapture from './PhotoCapture';
import AnalysisResults from './AnalysisResults';
import ReportView from './ReportView';
import BeforeAfter from './BeforeAfter';
import AnnotationEditor from './AnnotationEditor';

// Lazy-load heavy 3D components (Three.js) to avoid crashing the initial render
const FaceModel3D = lazy(() => import('./FaceModel3D'));
const SurgerySimulator = lazy(() => import('./SurgerySimulator'));

/**
 * FaceAnalysis — Main orchestrator component for the skin analysis module.
 * Manages the analysis workflow: Capture → Analyze → Results → Report.
 */

const TABS = {
    GUIDED: 'guided',
    CAPTURE: 'capture',
    RESULTS: 'results',
    THREE_D: '3d',
    SIMULATE: 'simulate',
    REPORT: 'report',
    COMPARE: 'compare',
    ANNOTATE: 'annotate',
};

export default function FaceAnalysis({ serverUrl, onBack }) {
    const [activeTab, setActiveTab] = useState(TABS.GUIDED);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [sessionId, setSessionId] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState(null);
    const [previousSessions, setPreviousSessions] = useState([]);
    const [patientInfo, setPatientInfo] = useState({ name: '', age: '', skin_type: '' });
    const [analysisElapsed, setAnalysisElapsed] = useState(0);
    const abortRef = useRef(null);
    const timerRef = useRef(null);

    const analysisUrl = import.meta.env.VITE_ANALYSIS_URL || serverUrl.replace(/:\d+/, ':8766');

    // ── Elapsed Timer ──────────────────────────────────────────────────────
    useEffect(() => {
        if (isAnalyzing) {
            setAnalysisElapsed(0);
            timerRef.current = setInterval(() => setAnalysisElapsed(s => s + 1), 1000);
        } else {
            if (timerRef.current) clearInterval(timerRef.current);
        }
        return () => { if (timerRef.current) clearInterval(timerRef.current); };
    }, [isAnalyzing]);

    // ── Guided Capture Complete ────────────────────────────────────────────

    const handleGuidedComplete = useCallback((photos, patient) => {
        if (patient) setPatientInfo(patient);
        analyzePhotos(photos);
    }, []);

    // ── API Calls ──────────────────────────────────────────────────────────

    const cancelAnalysis = useCallback(() => {
        if (abortRef.current) abortRef.current.abort();
        setIsAnalyzing(false);
        setError('Analysis cancelled');
    }, []);

    const analyzePhotos = useCallback(async (photos) => {
        setIsAnalyzing(true);
        setError(null);

        // Abort controller with 5-minute timeout
        const controller = new AbortController();
        abortRef.current = controller;
        const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

        try {
            const formData = new FormData();

            if (photos.length === 1) {
                formData.append('image', photos[0].blob, 'face.jpg');
                formData.append('angle', photos[0].angle || 'front_0');
                if (patientInfo.name) formData.append('patient_name', patientInfo.name);
                if (patientInfo.age) formData.append('patient_age', patientInfo.age);

                const res = await fetch(`${analysisUrl}/analyze`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                });

                if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
                const data = await res.json();
                setAnalysisResult(data);
                setSessionId(data.session_id);
                setActiveTab(TABS.RESULTS);
            } else {
                // Multi-angle analysis
                const angleMap = ['front', 'left_45', 'right_45', 'left_90', 'right_90'];
                photos.forEach((photo, i) => {
                    const fieldName = angleMap[i] || `angle_${i}`;
                    formData.append(fieldName, photo.blob, `${fieldName}.jpg`);
                });
                if (patientInfo.name) formData.append('patient_name', patientInfo.name);

                const res = await fetch(`${analysisUrl}/analyze/multi`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal,
                });

                if (!res.ok) throw new Error(`Multi-analysis failed: ${res.status}`);
                const data = await res.json();
                setAnalysisResult(data);
                setSessionId(data.session_id);
                setActiveTab(TABS.RESULTS);
            }

            // Refresh session list
            fetchSessions();
        } catch (err) {
            if (err.name === 'AbortError') {
                setError('Analysis timed out or was cancelled. Please try again.');
            } else {
                setError(err.message);
            }
        } finally {
            clearTimeout(timeoutId);
            abortRef.current = null;
            setIsAnalyzing(false);
        }
    }, [analysisUrl, patientInfo]);

    const fetchSessions = useCallback(async () => {
        try {
            const res = await fetch(`${analysisUrl}/sessions`);
            if (res.ok) {
                const data = await res.json();
                setPreviousSessions(data);
            }
        } catch { /* ignore */ }
    }, [analysisUrl]);

    const loadSession = useCallback(async (sid) => {
        try {
            const res = await fetch(`${analysisUrl}/sessions/${sid}`);
            if (res.ok) {
                const data = await res.json();
                setAnalysisResult(data);
                setSessionId(sid);
                setActiveTab(TABS.RESULTS);
            }
        } catch (err) {
            setError(err.message);
        }
    }, [analysisUrl]);

    // ── Render ─────────────────────────────────────────────────────────────

    const tabClass = (tab) =>
        `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeTab === tab
                ? 'bg-indigo-600 text-white shadow-lg'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
        }`;

    return (
        <div className="flex flex-col h-full bg-gray-950 text-white">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-3 border-b border-gray-800">
                <div className="flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="text-gray-400 hover:text-white text-sm"
                    >
                        ← Back
                    </button>
                    <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                        🏥 Face Analysis
                    </h1>
                </div>
                <div className="flex gap-2">
                    <button onClick={() => setActiveTab(TABS.GUIDED)} className={tabClass(TABS.GUIDED)}>
                        🧭 Guided Capture
                    </button>
                    <button onClick={() => setActiveTab(TABS.CAPTURE)} className={tabClass(TABS.CAPTURE)}>
                        📸 Quick Capture
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.RESULTS)}
                        className={tabClass(TABS.RESULTS)}
                        disabled={!analysisResult}
                    >
                        📊 Results
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.THREE_D)}
                        className={tabClass(TABS.THREE_D)}
                        disabled={!analysisResult}
                    >
                        🧊 3D Model
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.SIMULATE)}
                        className={tabClass(TABS.SIMULATE)}
                        disabled={!analysisResult}
                    >
                        🔬 Surgery Sim
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.ANNOTATE)}
                        className={tabClass(TABS.ANNOTATE)}
                        disabled={!analysisResult}
                    >
                        ✍️ Annotate
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.COMPARE)}
                        className={tabClass(TABS.COMPARE)}
                        disabled={previousSessions.length < 2}
                    >
                        📈 Compare
                    </button>
                    <button
                        onClick={() => setActiveTab(TABS.REPORT)}
                        className={tabClass(TABS.REPORT)}
                        disabled={!analysisResult}
                    >
                        📄 Report
                    </button>
                </div>
            </div>

            {/* Error Banner */}
            {error && (
                <div className="mx-6 mt-3 p-3 bg-red-900/50 border border-red-700 rounded-lg text-red-200 text-sm flex justify-between">
                    <span>⚠️ {error}</span>
                    <button onClick={() => setError(null)} className="text-red-400 hover:text-red-200">✕</button>
                </div>
            )}

            {/* Loading Overlay */}
            {isAnalyzing && (
                <div className="absolute inset-0 bg-black/70 flex items-center justify-center z-50">
                    <div className="bg-gray-900 rounded-2xl p-8 text-center shadow-2xl min-w-[320px]">
                        <div className="animate-spin w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-4" />
                        <p className="text-lg font-medium">Analyzing skin...</p>
                        <p className="text-gray-400 text-sm mt-1">Running 8 AI models across all photos</p>
                        <p className="text-indigo-400 text-sm mt-2 font-mono">
                            {Math.floor(analysisElapsed / 60)}:{String(analysisElapsed % 60).padStart(2, '0')} elapsed
                        </p>
                        {analysisElapsed > 10 && (
                            <p className="text-yellow-400/70 text-xs mt-1">Multi-angle analysis can take 1-3 minutes</p>
                        )}
                        <button
                            onClick={cancelAnalysis}
                            className="mt-4 px-4 py-2 bg-red-600/80 hover:bg-red-600 text-white text-sm rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            )}

            {/* Content */}
            <div className="flex-1 overflow-auto">
                {activeTab === TABS.GUIDED && (
                    <GuidedCapture
                        onComplete={handleGuidedComplete}
                        serverUrl={serverUrl}
                    />
                )}
                {activeTab === TABS.CAPTURE && (
                    <PhotoCapture
                        onAnalyze={analyzePhotos}
                        patientInfo={patientInfo}
                        onPatientChange={setPatientInfo}
                        previousSessions={previousSessions}
                        onLoadSession={loadSession}
                    />
                )}
                {activeTab === TABS.RESULTS && analysisResult && (
                    <AnalysisResults result={analysisResult} />
                )}
                {activeTab === TABS.THREE_D && analysisResult && (
                    <Suspense fallback={<div className="flex items-center justify-center h-full bg-gray-950 text-gray-400"><div className="text-center"><div className="animate-spin w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full mx-auto mb-3" /><p>Loading 3D Model...</p></div></div>}>
                        <FaceModel3D result={analysisResult} patientInfo={patientInfo} />
                    </Suspense>
                )}
                {activeTab === TABS.SIMULATE && analysisResult && (
                    <Suspense fallback={<div className="flex items-center justify-center h-full bg-gray-950 text-gray-400"><div className="text-center"><div className="animate-spin w-10 h-10 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-3" /><p>Loading Surgery Simulator...</p></div></div>}>
                        <SurgerySimulator result={analysisResult} />
                    </Suspense>
                )}
                {activeTab === TABS.ANNOTATE && analysisResult && (
                    <AnnotationEditor
                        result={analysisResult}
                        serverUrl={analysisUrl}
                        sessionId={sessionId}
                    />
                )}
                {activeTab === TABS.COMPARE && (
                    <BeforeAfter
                        sessions={previousSessions}
                        serverUrl={analysisUrl}
                        onLoadSession={loadSession}
                    />
                )}
                {activeTab === TABS.REPORT && analysisResult && (
                    <ReportView
                        sessionId={sessionId}
                        serverUrl={analysisUrl}
                        result={analysisResult}
                        patientInfo={patientInfo}
                    />
                )}
            </div>
        </div>
    );
}
