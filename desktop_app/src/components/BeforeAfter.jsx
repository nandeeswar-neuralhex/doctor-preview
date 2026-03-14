import React, { useState, useRef, useCallback } from 'react';

/**
 * BeforeAfter — AURA-style side-by-side face comparison with slider.
 * Shows two face images with a draggable divider, plus score delta.
 */

function scoreColor(s) { return s >= 85 ? '#22c55e' : s >= 70 ? '#84cc16' : s >= 50 ? '#eab308' : '#ef4444'; }

export default function BeforeAfter({ sessions, serverUrl, onLoadSession }) {
    const [beforeId, setBeforeId] = useState('');
    const [afterId, setAfterId] = useState('');
    const [comparison, setComparison] = useState(null);
    const [loading, setLoading] = useState(false);
    const [sliderPos, setSliderPos] = useState(50);
    const [dragging, setDragging] = useState(false);
    const containerRef = useRef(null);

    const compare = async () => {
        if (!beforeId || !afterId) return;
        setLoading(true);
        try {
            const res = await fetch(`${serverUrl}/compare/${beforeId}/${afterId}`);
            if (res.ok) setComparison(await res.json());
        } catch (err) { console.error(err); }
        finally { setLoading(false); }
    };

    const handleMove = useCallback((e) => {
        if (!dragging || !containerRef.current) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = ((e.clientX || e.touches?.[0]?.clientX || 0) - rect.left) / rect.width * 100;
        setSliderPos(Math.max(5, Math.min(95, x)));
    }, [dragging]);

    const beforeImg = comparison?.session_before?.original_image;
    const afterImg = comparison?.session_after?.original_image;
    const hasImages = beforeImg && afterImg;

    return (
        <div className="flex flex-col h-full bg-black text-white">
            {/* Top: Session selector */}
            <div className="px-5 py-3 bg-gray-950/80 backdrop-blur border-b border-gray-800/50">
                <div className="flex gap-3 items-end max-w-4xl mx-auto">
                    <div className="flex-1">
                        <label className="text-[9px] text-gray-500 block mb-1">BEFORE</label>
                        <select value={beforeId} onChange={e => setBeforeId(e.target.value)}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs focus:border-indigo-500 outline-none">
                            <option value="">Select session…</option>
                            {sessions.map(s => (
                                <option key={s.session_id} value={s.session_id}>
                                    {s.patient?.name || 'Patient'} — Score: {s.overall_score?.toFixed(0) || '?'}
                                </option>
                            ))}
                        </select>
                    </div>
                    <span className="text-gray-600 text-xl pb-2">→</span>
                    <div className="flex-1">
                        <label className="text-[9px] text-gray-500 block mb-1">AFTER</label>
                        <select value={afterId} onChange={e => setAfterId(e.target.value)}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs focus:border-indigo-500 outline-none">
                            <option value="">Select session…</option>
                            {sessions.map(s => (
                                <option key={s.session_id} value={s.session_id}>
                                    {s.patient?.name || 'Patient'} — Score: {s.overall_score?.toFixed(0) || '?'}
                                </option>
                            ))}
                        </select>
                    </div>
                    <button onClick={compare} disabled={!beforeId || !afterId || loading}
                        className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-xs font-semibold disabled:opacity-30 transition-all">
                        {loading ? '⏳' : '🔍 Compare'}
                    </button>
                </div>
            </div>

            {/* Center: Visual comparison */}
            <div className="flex-1 flex items-center justify-center p-4 overflow-hidden">
                {hasImages ? (
                    <div className="flex flex-col items-center gap-4 w-full max-w-5xl">
                        {/* Score delta banner */}
                        <div className="flex items-center gap-6 mb-2">
                            <div className="text-center">
                                <p className="text-2xl font-black" style={{ color: scoreColor(comparison.session_before?.overall_score || 0) }}>
                                    {(comparison.session_before?.overall_score || 0).toFixed(0)}
                                </p>
                                <p className="text-[9px] text-gray-500">Before</p>
                            </div>
                            <div className={`text-xl font-black ${comparison.overall_delta > 0 ? 'text-green-400' : comparison.overall_delta < 0 ? 'text-red-400' : 'text-gray-500'}`}>
                                {comparison.overall_delta > 0 ? '+' : ''}{comparison.overall_delta?.toFixed(1)}
                            </div>
                            <div className="text-center">
                                <p className="text-2xl font-black" style={{ color: scoreColor(comparison.session_after?.overall_score || 0) }}>
                                    {(comparison.session_after?.overall_score || 0).toFixed(0)}
                                </p>
                                <p className="text-[9px] text-gray-500">After</p>
                            </div>
                        </div>

                        {/* Image slider comparison */}
                        <div
                            ref={containerRef}
                            className="relative w-full max-w-3xl aspect-[3/4] rounded-2xl overflow-hidden border border-gray-700 select-none cursor-col-resize"
                            onMouseMove={handleMove}
                            onTouchMove={handleMove}
                            onMouseUp={() => setDragging(false)}
                            onMouseLeave={() => setDragging(false)}
                            onTouchEnd={() => setDragging(false)}
                        >
                            {/* After image (full, underneath) */}
                            <img src={`data:image/jpeg;base64,${afterImg}`} alt="After" className="absolute inset-0 w-full h-full object-cover" />

                            {/* Before image (clipped) */}
                            <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderPos}%` }}>
                                <img src={`data:image/jpeg;base64,${beforeImg}`} alt="Before"
                                    className="absolute inset-0 w-full h-full object-cover" style={{ minWidth: containerRef.current?.offsetWidth || '100%' }} />
                            </div>

                            {/* Slider line */}
                            <div className="absolute top-0 bottom-0 w-[3px] bg-white/80 shadow-lg" style={{ left: `${sliderPos}%`, transform: 'translateX(-50%)' }}
                                onMouseDown={() => setDragging(true)}
                                onTouchStart={() => setDragging(true)}
                            >
                                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 bg-white/90 rounded-full flex items-center justify-center shadow-xl cursor-col-resize">
                                    <span className="text-black text-sm font-bold">⇔</span>
                                </div>
                            </div>

                            {/* Labels */}
                            <div className="absolute top-3 left-3 bg-black/60 backdrop-blur px-2 py-1 rounded-lg text-[10px] font-semibold">Before</div>
                            <div className="absolute top-3 right-3 bg-black/60 backdrop-blur px-2 py-1 rounded-lg text-[10px] font-semibold">After</div>
                        </div>

                        {/* Improvement / Regression badges */}
                        {(comparison.improvements?.length > 0 || comparison.regressions?.length > 0) && (
                            <div className="flex gap-6 mt-2 text-xs">
                                {comparison.improvements?.length > 0 && (
                                    <div className="flex flex-wrap gap-1.5">
                                        {comparison.improvements.slice(0, 6).map((item, i) => (
                                            <span key={i} className="bg-green-900/40 border border-green-700/30 text-green-400 px-2 py-0.5 rounded-full text-[10px]">
                                                ↑ {item.zone?.replace(/_/g, ' ')} +{item.delta?.toFixed(0)}
                                            </span>
                                        ))}
                                    </div>
                                )}
                                {comparison.regressions?.length > 0 && (
                                    <div className="flex flex-wrap gap-1.5">
                                        {comparison.regressions.slice(0, 6).map((item, i) => (
                                            <span key={i} className="bg-red-900/40 border border-red-700/30 text-red-400 px-2 py-0.5 rounded-full text-[10px]">
                                                ↓ {item.zone?.replace(/_/g, ' ')} {item.delta?.toFixed(0)}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                ) : comparison && !hasImages ? (
                    /* Fallback: text-only comparison when no images */
                    <div className="max-w-lg mx-auto space-y-4">
                        <div className="bg-gray-900 rounded-2xl p-6 text-center border border-gray-800">
                            <p className="text-gray-400 text-sm mb-3">Overall Change</p>
                            <div className="flex items-center justify-center gap-6">
                                <div>
                                    <p className="text-3xl font-black" style={{ color: scoreColor(comparison.session_before?.overall_score || 0) }}>
                                        {(comparison.session_before?.overall_score || 0).toFixed(0)}
                                    </p>
                                    <p className="text-[10px] text-gray-500">Before</p>
                                </div>
                                <span className="text-3xl text-gray-600">→</span>
                                <div>
                                    <p className="text-3xl font-black" style={{ color: scoreColor(comparison.session_after?.overall_score || 0) }}>
                                        {(comparison.session_after?.overall_score || 0).toFixed(0)}
                                    </p>
                                    <p className="text-[10px] text-gray-500">After</p>
                                </div>
                                <div className={`text-2xl font-black ${comparison.overall_delta > 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {comparison.overall_delta > 0 ? '+' : ''}{comparison.overall_delta?.toFixed(1)}
                                </div>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="text-center text-gray-600">
                        <p className="text-5xl mb-3">📈</p>
                        <p className="text-sm">Select two sessions to compare results</p>
                        <p className="text-[10px] text-gray-700 mt-1">Track skin improvements over multiple visits</p>
                    </div>
                )}
            </div>
        </div>
    );
}
