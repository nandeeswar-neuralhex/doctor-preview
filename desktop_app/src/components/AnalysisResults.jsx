import React, { useState, useRef, useEffect, useCallback } from 'react';
import FaceMeasurements from './FaceMeasurements';

/**
 * AnalysisResults — AURA 3D-style visual analysis dashboard.
 *
 * Hero face image with interactive heatmap overlays, face mesh,
 * clickable zone score badges, and a sliding detail panel.
 */

const SCORE_COLORS = { excellent: '#22c55e', good: '#84cc16', fair: '#eab308', poor: '#ef4444' };
function scoreColor(s) { return s >= 85 ? SCORE_COLORS.excellent : s >= 70 ? SCORE_COLORS.good : s >= 50 ? SCORE_COLORS.fair : SCORE_COLORS.poor; }
function scoreLabel(s) { return s >= 85 ? 'Excellent' : s >= 70 ? 'Good' : s >= 50 ? 'Fair' : 'Needs Attention'; }

const MODES = [
    { id: 'original',     label: 'Original',     icon: '📷' },
    { id: 'wrinkle',      label: 'Wrinkles',     icon: '〰️', color: '#a855f7' },
    { id: 'pore',         label: 'Pores',        icon: '🔵', color: '#3b82f6' },
    { id: 'pigmentation', label: 'Pigmentation', icon: '🟤', color: '#f97316' },
    { id: 'redness',      label: 'Redness',      icon: '🔴', color: '#ef4444' },
    { id: 'texture',      label: 'Texture',      icon: '🟢', color: '#22c55e' },
    { id: 'mesh',         label: 'Face Mesh',    icon: '🕸️', color: '#22c55e' },
    { id: 'grid',         label: 'Measurements', icon: '📐', color: '#6366f1' },
];

/* approximate zone centres (% of face image) — aligned with backend ZONE_LANDMARKS */
const ZONE_POS = {
    forehead:        { x: 50, y: 14 },
    left_temple:     { x: 22, y: 18 },
    right_temple:    { x: 78, y: 18 },
    under_eye_left:  { x: 33, y: 34 },
    under_eye_right: { x: 67, y: 34 },
    nose:            { x: 50, y: 44 },
    left_cheek:      { x: 20, y: 55 },
    right_cheek:     { x: 80, y: 55 },
    jawline_left:    { x: 25, y: 72 },
    jawline_right:   { x: 75, y: 72 },
    chin:            { x: 50, y: 82 },
};

/* Face mesh contours (MediaPipe landmark indices) */
const CONTOURS = [
    [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],
    [70,63,105,66,107,55,65,52,53,46],
    [300,293,334,296,336,285,295,282,283,276],
    [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61],
    [168,6,197,195,5,4,1,19],
    [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33],
    [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362],
];

export default function AnalysisResults({ result }) {
    const [mode, setMode] = useState('original');
    const [opacity, setOpacity] = useState(0.55);
    const [showZones, setShowZones] = useState(true);
    const [showPanel, setShowPanel] = useState(true);
    const [selZone, setSelZone] = useState(null);
    const [hovZone, setHovZone] = useState(null);
    const [showMeasure, setShowMeasure] = useState(false);
    const canvasRef = useRef(null);
    const [faceImg, setFaceImg] = useState(null);
    const [hmImgs, setHmImgs] = useState({});

    if (!result) return null;

    const {
        overall_score = 0, skin_age, zone_scores = {}, conditions = [],
        symmetry = {}, measurements = {}, heatmaps = {}, recommendations = [],
        processing_time_ms = 0, analyzer_scores = {}, original_image, landmarks,
    } = result;

    /* load face */
    useEffect(() => {
        if (!original_image) return;
        const img = new Image();
        img.onload = () => setFaceImg(img);
        img.src = `data:image/jpeg;base64,${original_image}`;
    }, [original_image]);

    /* load heatmaps */
    useEffect(() => {
        Object.entries(heatmaps).forEach(([k, b64]) => {
            const img = new Image();
            img.onload = () => setHmImgs(prev => ({ ...prev, [k]: img }));
            img.src = `data:image/png;base64,${b64}`;
        });
    }, [heatmaps]);

    /* draw mesh */
    const drawMesh = useCallback((ctx, w, h) => {
        if (!landmarks?.length) return;
        // points
        ctx.fillStyle = 'rgba(34,197,94,0.7)';
        landmarks.forEach(p => { ctx.beginPath(); ctx.arc(p.x * w, p.y * h, 1.2, 0, Math.PI * 2); ctx.fill(); });
        // contour lines
        ctx.strokeStyle = 'rgba(34,197,94,0.35)';
        ctx.lineWidth = 0.8;
        CONTOURS.forEach(c => {
            if (c.some(i => i >= landmarks.length)) return;
            ctx.beginPath();
            c.forEach((idx, i) => { if (idx < landmarks.length) { const x = landmarks[idx].x * w, y = landmarks[idx].y * h; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); } });
            ctx.stroke();
        });
    }, [landmarks]);

    /* draw measurement grid */
    const drawGrid = useCallback((ctx, w, h) => {
        ctx.strokeStyle = 'rgba(99,102,241,0.4)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([4, 4]);
        // horizontal thirds
        [0.33, 0.5, 0.67].forEach(r => { ctx.beginPath(); ctx.moveTo(0, r * h); ctx.lineTo(w, r * h); ctx.stroke(); });
        // vertical thirds
        [0.33, 0.5, 0.67].forEach(r => { ctx.beginPath(); ctx.moveTo(r * w, 0); ctx.lineTo(r * w, h); ctx.stroke(); });
        ctx.setLineDash([]);

        // measurement labels from data
        if (measurements && landmarks?.length > 10) {
            ctx.font = '11px monospace';
            ctx.fillStyle = 'rgba(99,102,241,0.9)';
            // inter-pupillary
            if (landmarks[33] && landmarks[263]) {
                const lx = landmarks[33].x * w, ly = landmarks[33].y * h;
                const rx = landmarks[263].x * w, ry = landmarks[263].y * h;
                ctx.strokeStyle = 'rgba(129,140,248,0.7)';
                ctx.lineWidth = 1.5;
                ctx.setLineDash([]);
                ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(rx, ry); ctx.stroke();
                const dist = measurements.inter_pupillary_distance?.value;
                if (dist) ctx.fillText(`${dist}`, (lx + rx) / 2 - 10, ly - 8);
            }
            // nose length
            if (landmarks[168] && landmarks[1]) {
                const tx = landmarks[168].x * w, ty = landmarks[168].y * h;
                const bx = landmarks[1].x * w, by = landmarks[1].y * h;
                ctx.strokeStyle = 'rgba(129,140,248,0.7)';
                ctx.lineWidth = 1.5;
                ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(bx, by); ctx.stroke();
                const nose = measurements.nose_length?.value;
                if (nose) ctx.fillText(`${nose}`, bx + 8, (ty + by) / 2);
            }
        }
    }, [landmarks, measurements]);

    /* canvas render */
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !faceImg) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width = faceImg.width;
        const h = canvas.height = faceImg.height;
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(faceImg, 0, 0, w, h);

        if (mode !== 'original' && mode !== 'mesh' && mode !== 'grid') {
            const hm = hmImgs[mode];
            if (hm) { ctx.globalAlpha = opacity; ctx.drawImage(hm, 0, 0, w, h); ctx.globalAlpha = 1; }
        }
        if (mode === 'mesh') drawMesh(ctx, w, h);
        if (mode === 'grid') drawGrid(ctx, w, h);
    }, [faceImg, mode, opacity, hmImgs, drawMesh, drawGrid]);

    const zoneDetail = selZone && zone_scores[selZone];

    return (
        <div className="flex h-full bg-black">
            {/* ── LEFT: Score Panel ── */}
            {showPanel && (
                <div className="w-[260px] bg-gray-950/90 backdrop-blur-xl border-r border-gray-800/60 overflow-y-auto shrink-0">
                    {/* Overall score ring */}
                    <div className="p-5 text-center border-b border-gray-800/40">
                        <div className="relative inline-block">
                            <svg width="110" height="110" className="-rotate-90">
                                <circle cx="55" cy="55" r="46" fill="none" stroke="#1f2937" strokeWidth="7" />
                                <circle cx="55" cy="55" r="46" fill="none" stroke={scoreColor(overall_score)} strokeWidth="7" strokeLinecap="round" strokeDasharray={`${overall_score * 2.89} 289`} />
                            </svg>
                            <div className="absolute inset-0 flex flex-col items-center justify-center">
                                <span className="text-2xl font-black" style={{ color: scoreColor(overall_score) }}>{overall_score.toFixed(0)}</span>
                                <span className="text-[8px] text-gray-600">/100</span>
                            </div>
                        </div>
                        <p className="text-xs font-semibold mt-2" style={{ color: scoreColor(overall_score) }}>{scoreLabel(overall_score)}</p>
                        {skin_age && <p className="text-[10px] text-gray-500 mt-0.5">Skin Age: <b className="text-white">{skin_age}</b></p>}
                    </div>

                    {/* Analyzer bars */}
                    <div className="px-4 py-3 border-b border-gray-800/40">
                        <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Breakdown</h4>
                        {Object.entries(analyzer_scores).map(([k, v]) => (
                            <div key={k} className="mb-2">
                                <div className="flex justify-between text-[10px] mb-0.5">
                                    <span className="text-gray-400 capitalize">{k.replace(/_/g, ' ')}</span>
                                    <span className="font-bold" style={{ color: scoreColor(v) }}>{v.toFixed(0)}</span>
                                </div>
                                <div className="h-[5px] bg-gray-800 rounded-full overflow-hidden">
                                    <div className="h-full rounded-full transition-all duration-700" style={{ width: `${v}%`, background: scoreColor(v) }} />
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Symmetry */}
                    {symmetry?.overall_score > 0 && (
                        <div className="px-4 py-3 border-b border-gray-800/40">
                            <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Symmetry</h4>
                            {[['Overall', symmetry.overall_score], ['Eyes', symmetry.eye_alignment], ['Cheeks', symmetry.cheek_balance], ['Jaw', symmetry.jawline_symmetry], ['Lips', symmetry.lip_symmetry]].map(([l, v]) => v > 0 && (
                                <div key={l} className="flex items-center gap-1.5 mb-1">
                                    <span className="w-12 text-[9px] text-gray-500">{l}</span>
                                    <div className="flex-1 h-[4px] bg-gray-800 rounded-full overflow-hidden">
                                        <div className="h-full rounded-full" style={{ width: `${v}%`, background: scoreColor(v) }} />
                                    </div>
                                    <span className="w-7 text-right text-[9px] font-bold" style={{ color: scoreColor(v) }}>{v.toFixed(0)}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Conditions */}
                    {conditions.length > 0 && (
                        <div className="px-4 py-3 border-b border-gray-800/40">
                            <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Conditions ({conditions.length})</h4>
                            {conditions.slice(0, 10).map((c, i) => (
                                <div key={i} className="flex items-center gap-1.5 mb-1 text-[10px]">
                                    <div className={`w-1.5 h-1.5 rounded-full ${c.severity === 'severe' ? 'bg-red-500' : c.severity === 'moderate' ? 'bg-yellow-500' : 'bg-green-500'}`} />
                                    <span className="text-gray-300 capitalize truncate">{c.type?.replace(/_/g, ' ')}</span>
                                    <span className="ml-auto text-gray-600 text-[8px] capitalize">{c.zone?.replace(/_/g, ' ')}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Recommendations */}
                    {recommendations.length > 0 && (
                        <div className="px-4 py-3">
                            <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Recommendations</h4>
                            {recommendations.slice(0, 5).map((r, i) => (
                                <div key={i} className="bg-indigo-950/40 border border-indigo-800/20 rounded-lg p-2 mb-1.5">
                                    <p className="text-[10px] font-medium text-gray-300">{r.area}</p>
                                    <p className="text-[9px] text-gray-500 mt-0.5 leading-tight">{r.suggestion}</p>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            {/* ── CENTER: Face Visual ── */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Mode toolbar */}
                <div className="flex items-center gap-1.5 px-3 py-2 bg-gray-950/80 backdrop-blur border-b border-gray-800/40 overflow-x-auto">
                    {MODES.map(m => (
                        <button key={m.id} onClick={() => setMode(m.id)}
                            className={`shrink-0 px-2.5 py-1.5 rounded-lg text-[11px] font-medium flex items-center gap-1 transition-all
                                ${mode === m.id ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/20' : 'bg-gray-800/60 text-gray-400 hover:bg-gray-700 hover:text-white'}`}
                        >
                            <span>{m.icon}</span><span className="hidden sm:inline">{m.label}</span>
                        </button>
                    ))}
                    <div className="flex-1" />
                    {mode !== 'original' && mode !== 'mesh' && mode !== 'grid' && (
                        <div className="flex items-center gap-1.5 shrink-0">
                            <span className="text-[9px] text-gray-600">Opacity</span>
                            <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={e => setOpacity(+e.target.value)} className="w-16 h-1 accent-indigo-500" />
                        </div>
                    )}
                    <button onClick={() => setShowZones(!showZones)} className={`shrink-0 px-2 py-1 rounded text-[9px] ${showZones ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-500'}`}>Zones</button>
                    <button onClick={() => setShowPanel(!showPanel)} className={`shrink-0 px-2 py-1 rounded text-[9px] ${showPanel ? 'bg-indigo-600 text-white' : 'bg-gray-800 text-gray-500'}`}>Panel</button>
                </div>

                {/* Face canvas area */}
                <div className="flex-1 flex items-center justify-center relative overflow-hidden bg-black">
                    {faceImg ? (
                        <div className="relative inline-block max-h-full max-w-full">
                            <canvas ref={canvasRef} className="max-h-[calc(100vh-160px)] max-w-full object-contain" />

                            {/* Zone badges */}
                            {showZones && Object.entries(ZONE_POS).map(([zone, pos]) => {
                                const zd = zone_scores[zone];
                                if (!zd) return null;
                                const s = zd.overall || 0;
                                const active = hovZone === zone || selZone === zone;
                                return (
                                    <div key={zone}
                                        className="absolute cursor-pointer transition-all duration-150"
                                        style={{ left: `${pos.x}%`, top: `${pos.y}%`, transform: `translate(-50%,-50%) scale(${active ? 1.3 : 1})`, zIndex: active ? 30 : 10 }}
                                        onMouseEnter={() => setHovZone(zone)}
                                        onMouseLeave={() => setHovZone(null)}
                                        onClick={() => setSelZone(selZone === zone ? null : zone)}
                                    >
                                        <div className={`flex items-center gap-1 px-2 py-[3px] rounded-full backdrop-blur-md border transition-all
                                            ${selZone === zone ? 'bg-indigo-900/90 border-indigo-400 shadow-lg shadow-indigo-500/40' : 'bg-black/55 border-gray-600/40 hover:border-white/40'}`}>
                                            <div className="w-2 h-2 rounded-full" style={{ background: scoreColor(s) }} />
                                            <span className="text-[8px] font-semibold text-white/90 capitalize whitespace-nowrap">{zone.replace(/_/g, ' ')}</span>
                                            <span className="text-[9px] font-black" style={{ color: scoreColor(s) }}>{s.toFixed(0)}</span>
                                        </div>
                                    </div>
                                );
                            })}

                            {/* Mini thumbnail when overlay active */}
                            {mode !== 'original' && original_image && (
                                <div className="absolute bottom-3 left-3 w-24 h-24 rounded-xl overflow-hidden border border-gray-600/60 shadow-2xl cursor-pointer hover:scale-105 transition-transform"
                                    onClick={() => setMode('original')}
                                >
                                    <img src={`data:image/jpeg;base64,${original_image}`} alt="" className="w-full h-full object-cover" />
                                    <div className="absolute bottom-0 inset-x-0 bg-black/70 text-[7px] text-center py-0.5 text-gray-400">Original</div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="text-center text-gray-600">
                            <p className="text-5xl mb-3">🔬</p>
                            <p className="text-sm">Run an analysis to see visual results</p>
                        </div>
                    )}
                </div>

                {/* Bottom bar */}
                <div className="flex items-center justify-between px-4 py-1.5 bg-gray-950/80 backdrop-blur border-t border-gray-800/40 text-[9px] text-gray-600">
                    <div className="flex gap-4">
                        <span>⏱ {processing_time_ms.toFixed(0)}ms</span>
                        <span>🔬 8 Analyzers</span>
                        <span>📊 {Object.keys(zone_scores).length} Zones</span>
                        <span>🎯 {conditions.length} Conditions</span>
                    </div>
                    {mode !== 'original' && <span className="text-indigo-400">Viewing: {mode}</span>}
                </div>
            </div>

            {/* ── RIGHT: Zone Detail / Measurements ── */}
            {selZone && zoneDetail && (
                <div className="w-[240px] bg-gray-950/90 backdrop-blur-xl border-l border-gray-800/60 p-4 overflow-y-auto shrink-0">
                    <div className="flex justify-between items-center mb-3">
                        <h3 className="text-sm font-bold text-white capitalize">{selZone.replace(/_/g, ' ')}</h3>
                        <button onClick={() => setSelZone(null)} className="text-gray-600 hover:text-white">✕</button>
                    </div>
                    <div className="text-center mb-4">
                        <span className="text-3xl font-black" style={{ color: scoreColor(zoneDetail.overall || 0) }}>{(zoneDetail.overall || 0).toFixed(0)}</span>
                        <p className="text-[9px] text-gray-600 mt-1">Zone Score</p>
                    </div>
                    {['wrinkle', 'pore', 'pigmentation', 'redness', 'texture'].map(m => {
                        const v = zoneDetail[m] || 0;
                        return (
                            <div key={m} className="mb-2.5">
                                <div className="flex justify-between text-[10px] mb-0.5">
                                    <span className="text-gray-400 capitalize">{m}</span>
                                    <span className="font-bold" style={{ color: scoreColor(v) }}>{v.toFixed(0)}</span>
                                </div>
                                <div className="h-[5px] bg-gray-800 rounded-full overflow-hidden">
                                    <div className="h-full rounded-full" style={{ width: `${v}%`, background: scoreColor(v) }} />
                                </div>
                            </div>
                        );
                    })}
                    {conditions.filter(c => c.zone === selZone).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-800/40">
                            <h4 className="text-[9px] font-bold text-gray-500 uppercase mb-1.5">Zone Issues</h4>
                            {conditions.filter(c => c.zone === selZone).map((c, i) => (
                                <div key={i} className="flex items-center gap-1.5 mb-1 text-[10px]">
                                    <div className={`w-1.5 h-1.5 rounded-full ${c.severity === 'severe' ? 'bg-red-500' : c.severity === 'moderate' ? 'bg-yellow-500' : 'bg-green-500'}`} />
                                    <span className="text-gray-300 capitalize">{c.type?.replace(/_/g, ' ')}</span>
                                </div>
                            ))}
                        </div>
                    )}
                    {/* Measurements toggle */}
                    {Object.keys(measurements).length > 0 && (
                        <div className="mt-3 pt-3 border-t border-gray-800/40">
                            <button onClick={() => setShowMeasure(!showMeasure)} className="text-[9px] text-indigo-400 hover:text-indigo-300">{showMeasure ? '▼' : '▶'} Measurements</button>
                            {showMeasure && <FaceMeasurements measurements={measurements} />}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
