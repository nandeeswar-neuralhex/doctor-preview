import React, { useState, useRef, useCallback, useEffect } from 'react';

/**
 * ReportView — AURA-quality professional Face Analysis Report.
 *
 * Multi-page printable report with:
 *   - Cover page (wireframe face, patient info, clinic branding)
 *   - Summary page (overall score, skin age, breakdown bars)
 *   - Skin Analysis pages (Wrinkles, Texture, Brown Spots, Red Areas, Pores)
 *     each with regional score face diagram + severity explanation
 *   - Conditions & Recommendations page
 *   - Doctor's notes page
 */

const SEVERITY_COLORS = ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444'];
const scoreColor = (s) => s >= 85 ? '#22c55e' : s >= 70 ? '#84cc16' : s >= 50 ? '#eab308' : '#ef4444';
const toAura = (s) => (5 - Math.max(0, Math.min(100, s)) / 25).toFixed(1);

const SKIN_TYPES = [
    { id: 'wrinkles',    label: 'WRINKLES',     desc: 'Wrinkles are creases, folds, or ridges in the skin that typically appear as the skin ages. They result from the gradual loss of collagen and elastin, which are proteins that help keep the skin supple and firm.' },
    { id: 'texture',     label: 'TEXTURE',      desc: 'Skin texture analysis evaluates the smoothness and consistency of the skin surface. Uneven texture can result from dead skin cell buildup, sun damage, aging, or skin conditions.' },
    { id: 'brown_spots', label: 'BROWN SPOTS',  desc: 'Brown spots (also known as age spots, liver spots, or solar lentigines) are flat, tan to dark brown spots on the skin. They are caused by overactive pigment cells triggered by UV exposure.' },
    { id: 'red_areas',   label: 'RED AREAS',    desc: 'Red areas on the face refer to areas of the skin that appear red or flushed in color. Various factors, including skin conditions, prominent or dilated blood vessels, inflammation, and external influences, that can cause redness or inflammation.' },
    { id: 'pores',       label: 'PORES',        desc: 'Pores are tiny openings in the skin that release oils and sweat. While pores serve essential functions, enlarged or visible pores can affect skin appearance. Factors include genetics, age, sun damage, and excess oil production.' },
];

/* ── Wireframe face SVG for cover page ─────────────────────────── */
function WireframeFace() {
    return (
        <svg viewBox="0 0 300 400" className="w-full h-full" fill="none" stroke="#8b7d6b" strokeWidth="0.5" opacity="0.7">
            {/* Face outline */}
            <ellipse cx="150" cy="190" rx="100" ry="130" />
            {/* Forehead lines */}
            {[145, 160, 175, 190, 205, 220].map(y => (
                <path key={y} d={`M${80 + (y - 145) * 0.3} ${y} Q150 ${y - 8} ${220 - (y - 145) * 0.3} ${y}`} />
            ))}
            {/* Eye sockets */}
            <ellipse cx="115" cy="185" rx="25" ry="12" />
            <ellipse cx="185" cy="185" rx="25" ry="12" />
            {/* Eyebrows */}
            <path d="M85 170 Q115 158 145 168" />
            <path d="M155 168 Q185 158 215 170" />
            {/* Nose */}
            <path d="M150 175 L150 230" />
            <path d="M135 230 Q150 240 165 230" />
            {/* Nose bridge lines */}
            <path d="M140 190 L138 225" />
            <path d="M160 190 L162 225" />
            {/* Mouth */}
            <path d="M120 260 Q150 275 180 260" />
            <path d="M125 260 Q150 252 175 260" />
            {/* Chin */}
            <path d="M110 280 Q150 320 190 280" />
            {/* Jaw lines */}
            <path d="M55 200 Q60 280 110 300" />
            <path d="M245 200 Q240 280 190 300" />
            {/* Grid lines (topological) */}
            {[0, 1, 2, 3, 4, 5, 6].map(i => {
                const y = 140 + i * 30;
                const w = 100 - Math.abs(i - 3) * 10;
                return <line key={`h${i}`} x1={150 - w} y1={y} x2={150 + w} y2={y} strokeDasharray="3 5" opacity="0.3" />;
            })}
            {[0, 1, 2, 3, 4].map(i => {
                const x = 100 + i * 25;
                return <line key={`v${i}`} x1={x} y1={130} x2={x} y2={310} strokeDasharray="3 5" opacity="0.3" />;
            })}
            {/* Cheek contour lines */}
            <path d="M70 195 Q85 240 110 270" strokeDasharray="2 4" opacity="0.4" />
            <path d="M230 195 Q215 240 190 270" strokeDasharray="2 4" opacity="0.4" />
            {/* Ear outlines */}
            <path d="M50 170 Q40 190 45 215 Q50 230 55 220" />
            <path d="M250 170 Q260 190 255 215 Q250 230 245 220" />
        </svg>
    );
}

/* ── Regional Score Mini Diagram (for report pages) ────────────── */
function ReportRegionalDiagram({ zoneScores, analysisType }) {
    const zColor = (s) => (s || 70) >= 80 ? '#22c55e' : (s || 70) >= 60 ? '#a3e635' : (s || 70) >= 40 ? '#eab308' : '#f97316';
    const zones = [
        { id: 'forehead',    x: 50, y: 18, score: zoneScores?.forehead?.overall },
        { id: 'left_cheek',  x: 22, y: 52, score: zoneScores?.left_cheek?.overall },
        { id: 'nose',        x: 50, y: 45, score: zoneScores?.nose?.overall },
        { id: 'right_cheek', x: 78, y: 52, score: zoneScores?.right_cheek?.overall },
    ];

    return (
        <div className="relative w-32 h-40">
            <svg viewBox="0 0 100 120" className="w-full h-full absolute inset-0 opacity-25">
                <ellipse cx="50" cy="55" rx="38" ry="50" fill="#555" />
                <ellipse cx="35" cy="42" rx="8" ry="5" fill="#333" />
                <ellipse cx="65" cy="42" rx="8" ry="5" fill="#333" />
            </svg>
            {zones.map(z => (
                <div key={z.id} className="absolute flex flex-col items-center" style={{ left: `${z.x}%`, top: `${z.y}%`, transform: 'translate(-50%,-50%)' }}>
                    <div className="w-9 h-7 rounded" style={{ background: zColor(z.score), opacity: 0.6 }} />
                    <span className="text-[9px] font-bold text-gray-800">{toAura(z.score)}</span>
                </div>
            ))}
        </div>
    );
}

/* ── Overall Score Diagram ─────────────────────────────────────── */
function OverallScoreDiagram({ score, label }) {
    const col = scoreColor(score);
    return (
        <div className="relative w-28 h-28 flex items-center justify-center">
            <svg viewBox="0 0 100 100" className="w-full h-full absolute inset-0 -rotate-90">
                <circle cx="50" cy="50" r="42" fill="none" stroke="#e5e7eb" strokeWidth="6" />
                <circle cx="50" cy="50" r="42" fill="none" stroke={col} strokeWidth="6" strokeLinecap="round"
                    strokeDasharray={`${score * 2.64} 264`} />
            </svg>
            <div className="flex flex-col items-center z-10">
                <span className="text-lg font-black" style={{ color: col }}>{toAura(score)}</span>
                {label && <span className="text-[7px] text-gray-500 font-medium">{label}</span>}
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN REPORT VIEW
   ═══════════════════════════════════════════════════════════════════ */
export default function ReportView({ sessionId, serverUrl, result, patientInfo }) {
    const [doctorNotes, setDoctorNotes] = useState('');
    const [currentPage, setCurrentPage] = useState(0);
    const reportRef = useRef(null);

    if (!result) return null;

    const {
        overall_score = 0, skin_age, zone_scores = {}, conditions = [],
        symmetry = {}, measurements = {}, recommendations = [],
        processing_time_ms = 0, analyzer_scores = {}, original_image, landmarks,
    } = result;

    const pName = patientInfo?.name || 'Patient';
    const pAge = patientInfo?.age || '';
    const dateStr = new Date().toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' });

    // Total pages: cover + summary + 5 skin types + conditions + notes
    const totalPages = 9;

    const printReport = useCallback(() => {
        const el = reportRef.current;
        if (!el) return;
        const win = window.open('', '_blank');
        win.document.write(`<!DOCTYPE html><html><head><title>Face Analysis Report — ${pName}</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');
                body { margin: 0; font-family: 'Inter', sans-serif; background: #fff; color: #1a1a1a; }
                @page { size: A4; margin: 0; }
                @media print { .no-print { display: none !important; } .page-break { page-break-after: always; } }
            </style></head><body>${el.innerHTML}</body></html>`);
        win.document.close();
        setTimeout(() => { win.focus(); win.print(); }, 500);
    }, [pName]);

    /* ── Page definitions ── */
    const pages = [
        /* 0: COVER */
        <div key="cover" className="w-full h-full bg-white flex flex-col items-center justify-center relative p-12">
            <div className="absolute top-8 left-10 text-left">
                <p className="text-[11px] text-gray-500">{pName}</p>
                <p className="text-[9px] text-gray-400">{dateStr}</p>
            </div>
            <div className="w-48 h-64 mb-8 opacity-70">
                <WireframeFace />
            </div>
            <h1 className="text-4xl font-light text-gray-800 tracking-wide" style={{ fontFamily: 'Georgia, serif' }}>
                Face
            </h1>
            <h1 className="text-4xl font-light text-gray-800 tracking-wide" style={{ fontFamily: 'Georgia, serif' }}>
                Analysis
            </h1>
            <h1 className="text-4xl font-light text-gray-800 tracking-wide mb-10" style={{ fontFamily: 'Georgia, serif' }}>
                Report
            </h1>
            <div className="text-center mt-auto">
                <p className="text-[11px] font-semibold text-gray-700 tracking-widest">Beauty Clinic</p>
                <p className="text-[8px] text-gray-400 mt-1">Professional Skin Analysis System</p>
            </div>
            <div className="absolute bottom-6 right-8">
                <p className="text-[9px] text-gray-400 italic tracking-widest">doctor preview</p>
            </div>
        </div>,

        /* 1: SUMMARY */
        <div key="summary" className="w-full h-full bg-white p-10">
            <div className="flex items-start justify-between mb-8">
                <div>
                    <p className="text-[9px] text-gray-400 uppercase tracking-widest mb-1">Patient Summary</p>
                    <h2 className="text-2xl font-semibold text-gray-800" style={{ fontFamily: 'Georgia, serif' }}>{pName}</h2>
                    <p className="text-sm text-gray-500 mt-0.5">Age: {pAge} · Skin Age: {skin_age || '—'} · {dateStr}</p>
                </div>
                <OverallScoreDiagram score={overall_score} label="Overall" />
            </div>

            {/* Analyzer breakdown */}
            <div className="mb-8">
                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">Analysis Breakdown</h3>
                <div className="grid grid-cols-2 gap-x-8 gap-y-3">
                    {Object.entries(analyzer_scores).map(([k, v]) => (
                        <div key={k}>
                            <div className="flex justify-between text-[11px] mb-1">
                                <span className="text-gray-600 capitalize">{k.replace(/_/g, ' ')}</span>
                                <span className="font-bold" style={{ color: scoreColor(v) }}>{v.toFixed(0)}</span>
                            </div>
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                                <div className="h-full rounded-full transition-all" style={{ width: `${v}%`, background: scoreColor(v) }} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Symmetry */}
            {symmetry?.overall_score > 0 && (
                <div className="mb-8">
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">Facial Symmetry</h3>
                    <div className="grid grid-cols-3 gap-4">
                        {[['Overall', symmetry.overall_score], ['Eyes', symmetry.eye_alignment], ['Cheeks', symmetry.cheek_balance], ['Jaw', symmetry.jawline_symmetry], ['Lips', symmetry.lip_symmetry]].map(([l, v]) => v > 0 && (
                            <div key={l} className="text-center">
                                <span className="text-lg font-bold" style={{ color: scoreColor(v) }}>{v.toFixed(0)}</span>
                                <p className="text-[9px] text-gray-500">{l}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Zone scores grid */}
            <div>
                <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">Zone Scores</h3>
                <div className="grid grid-cols-4 gap-3">
                    {Object.entries(zone_scores).map(([zone, data]) => {
                        const s = data?.overall || 0;
                        return (
                            <div key={zone} className="text-center border border-gray-100 rounded-lg p-2">
                                <span className="text-sm font-bold" style={{ color: scoreColor(s) }}>{s.toFixed(0)}</span>
                                <p className="text-[8px] text-gray-500 capitalize mt-0.5">{zone.replace(/_/g, ' ')}</p>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="absolute bottom-6 right-8 text-[8px] text-gray-300">01</div>
        </div>,

        /* 2-6: Skin Analysis Pages */
        ...SKIN_TYPES.map((type, idx) => (
            <div key={type.id} className="w-full h-full bg-white p-10 relative">
                <p className="text-[8px] text-gray-400 uppercase tracking-widest mb-1">Skin Analysis</p>
                <h2 className="text-xl font-bold text-gray-800 mb-4 tracking-wide">{type.label}</h2>

                <p className="text-[11px] text-gray-600 leading-relaxed mb-6 max-w-lg">{type.desc}</p>

                <div className="flex gap-8 mb-6">
                    {/* Regional diagram */}
                    <div className="text-center">
                        <p className="text-[9px] text-gray-500 font-semibold mb-2">regional</p>
                        <ReportRegionalDiagram zoneScores={zone_scores} analysisType={type.id} />
                    </div>

                    {/* Overall for this type */}
                    <div className="text-center">
                        <p className="text-[9px] text-gray-500 font-semibold mb-2">overall</p>
                        <OverallScoreDiagram score={analyzer_scores[type.id] || analyzer_scores[type.id.replace('_', '')] || overall_score} label={type.label} />
                    </div>
                </div>

                {/* Severity scale */}
                <div className="flex items-center gap-1 mb-4 max-w-xs">
                    <span className="text-[7px] text-gray-400 mr-1">mild</span>
                    {SEVERITY_COLORS.map((c, i) => (
                        <div key={i} className="flex-1 h-4 rounded-sm flex items-center justify-center text-[8px] font-bold text-white" style={{ background: c }}>
                            {i + 1}
                        </div>
                    ))}
                    <span className="text-[7px] text-red-400 ml-1">severe</span>
                </div>

                {/* Zone detail for this analysis type */}
                <div className="grid grid-cols-2 gap-3 mt-4">
                    {Object.entries(zone_scores).slice(0, 8).map(([zone, data]) => {
                        const val = data?.[type.id] || data?.overall || 70;
                        return (
                            <div key={zone} className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full" style={{ background: scoreColor(val) }} />
                                <span className="text-[10px] text-gray-600 capitalize flex-1">{zone.replace(/_/g, ' ')}</span>
                                <span className="text-[10px] font-bold" style={{ color: scoreColor(val) }}>{val.toFixed(0)}</span>
                            </div>
                        );
                    })}
                </div>

                {/* 3D multi-angle face views (AURA-style: left profile + front + right profile) */}
                {original_image && (
                    <div className="absolute bottom-6 left-10 right-10 flex items-end justify-center gap-4">
                        {/* Left profile */}
                        <div className="w-28 h-32 rounded-xl overflow-hidden shadow-lg border border-gray-100 bg-gray-50 opacity-90"
                            style={{ transform: 'perspective(400px) rotateY(25deg)' }}>
                            <img src={`data:image/jpeg;base64,${original_image}`} alt="Left profile" className="w-full h-full object-cover"
                                style={{ filter: type.id === 'red_areas' ? 'saturate(2) contrast(1.3)' : type.id === 'texture' ? 'grayscale(1) contrast(1.5)' : 'none', objectPosition: '30% center' }} />
                        </div>
                        {/* Front view (larger) */}
                        <div className="w-32 h-36 rounded-xl overflow-hidden shadow-xl border border-gray-100 bg-gray-50 -mb-1">
                            <img src={`data:image/jpeg;base64,${original_image}`} alt="Front view" className="w-full h-full object-cover"
                                style={{ filter: type.id === 'red_areas' ? 'saturate(2) contrast(1.3)' : type.id === 'texture' ? 'grayscale(1) contrast(1.5)' : 'none' }} />
                        </div>
                        {/* Right profile */}
                        <div className="w-28 h-32 rounded-xl overflow-hidden shadow-lg border border-gray-100 bg-gray-50 opacity-90"
                            style={{ transform: 'perspective(400px) rotateY(-25deg)' }}>
                            <img src={`data:image/jpeg;base64,${original_image}`} alt="Right profile" className="w-full h-full object-cover"
                                style={{ filter: type.id === 'red_areas' ? 'saturate(2) contrast(1.3)' : type.id === 'texture' ? 'grayscale(1) contrast(1.5)' : 'none', objectPosition: '70% center' }} />
                        </div>
                        {/* Shadow/ground effect */}
                        <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-[70%] h-3 bg-gray-200/40 rounded-[100%] blur-sm" />
                    </div>
                )}

                <div className="absolute bottom-6 right-8 text-[8px] text-gray-300">{String(idx + 2).padStart(2, '0')}</div>
            </div>
        )),

        /* 7: CONDITIONS & RECOMMENDATIONS */
        <div key="conditions" className="w-full h-full bg-white p-10 relative">
            <p className="text-[8px] text-gray-400 uppercase tracking-widest mb-1">Clinical Report</p>
            <h2 className="text-xl font-bold text-gray-800 mb-6 tracking-wide">CONDITIONS & RECOMMENDATIONS</h2>

            <div className="grid grid-cols-2 gap-8">
                {/* Conditions */}
                <div>
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">
                        Detected Conditions ({conditions.length})
                    </h3>
                    <div className="space-y-2">
                        {conditions.slice(0, 12).map((c, i) => (
                            <div key={i} className="flex items-start gap-2">
                                <div className={`w-2 h-2 rounded-full mt-1 shrink-0 ${
                                    c.severity === 'severe' ? 'bg-red-500' : c.severity === 'moderate' ? 'bg-yellow-500' : 'bg-green-500'
                                }`} />
                                <div>
                                    <p className="text-[10px] font-medium text-gray-700 capitalize">{c.type?.replace(/_/g, ' ')}</p>
                                    <p className="text-[8px] text-gray-400 capitalize">{c.zone?.replace(/_/g, ' ')} · {c.severity}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Recommendations */}
                <div>
                    <h3 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-3 border-b border-gray-200 pb-1">
                        Treatment Recommendations
                    </h3>
                    <div className="space-y-3">
                        {recommendations.slice(0, 8).map((r, i) => (
                            <div key={i} className="border-l-2 pl-3" style={{ borderColor: scoreColor(50) }}>
                                <p className="text-[10px] font-semibold text-gray-700">{r.area}</p>
                                <p className="text-[9px] text-gray-500 leading-relaxed">{r.suggestion}</p>
                                {r.priority && <span className="text-[7px] uppercase tracking-wider text-orange-500 font-bold">{r.priority}</span>}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="absolute bottom-6 right-8 text-[8px] text-gray-300">07</div>
        </div>,

        /* 8: DOCTOR'S NOTES */
        <div key="notes" className="w-full h-full bg-white p-10 relative">
            <p className="text-[8px] text-gray-400 uppercase tracking-widest mb-1">Clinical Notes</p>
            <h2 className="text-xl font-bold text-gray-800 mb-6 tracking-wide">DOCTOR'S NOTES</h2>

            <div className="border border-gray-200 rounded-xl p-6 min-h-[300px]">
                {doctorNotes ? (
                    <p className="text-[11px] text-gray-700 leading-relaxed whitespace-pre-wrap">{doctorNotes}</p>
                ) : (
                    <p className="text-[11px] text-gray-400 italic">No clinical notes provided. Add notes in the editor panel.</p>
                )}
            </div>

            <div className="mt-10 pt-4 border-t border-gray-200 flex justify-between">
                <div>
                    <p className="text-[9px] text-gray-500">Doctor's Signature</p>
                    <div className="w-40 h-px bg-gray-300 mt-8" />
                </div>
                <div className="text-right">
                    <p className="text-[9px] text-gray-500">Date</p>
                    <p className="text-[10px] text-gray-600 mt-1">{dateStr}</p>
                </div>
            </div>

            <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
                <p className="text-[8px] text-gray-400 italic tracking-widest">doctor preview</p>
            </div>
            <div className="absolute bottom-6 right-8 text-[8px] text-gray-300">08</div>
        </div>,
    ];

    return (
        <div className="flex h-full bg-gray-950">
            {/* ── Left: Controls Panel ── */}
            <div className="w-80 bg-gray-950 border-r border-gray-800 p-5 flex flex-col gap-5 shrink-0 overflow-y-auto">
                {/* Header */}
                <div className="text-center pb-3 border-b border-gray-800/60">
                    <h3 className="text-sm font-bold text-white mb-0.5">📄 Professional Report</h3>
                    <p className="text-[9px] text-gray-500">AURA-quality PDF export</p>
                </div>

                {/* Page navigator */}
                <div>
                    <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Pages ({totalPages})</h4>
                    <div className="space-y-1">
                        {['Cover Page', 'Patient Summary', ...SKIN_TYPES.map(t => t.label), 'Conditions & Recs', 'Doctor Notes'].map((label, i) => (
                            <button key={i} onClick={() => setCurrentPage(i)}
                                className={`w-full text-left px-3 py-1.5 rounded-lg text-[10px] transition-all ${
                                    currentPage === i ? 'bg-indigo-600 text-white' : 'text-gray-400 hover:bg-gray-800'
                                }`}>
                                <span className="text-gray-500 mr-2 font-mono">{String(i + 1).padStart(2, '0')}</span>
                                {label}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Doctor's notes */}
                <div>
                    <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">📝 Doctor's Notes</h4>
                    <textarea
                        value={doctorNotes}
                        onChange={e => setDoctorNotes(e.target.value)}
                        placeholder="Add clinical observations, treatment plan, follow-up notes..."
                        className="w-full h-32 bg-gray-800 border border-gray-700 rounded-lg p-2.5 text-[11px] text-white resize-none focus:border-indigo-500 focus:outline-none"
                    />
                </div>

                {/* Quick Summary */}
                <div className="pt-3 border-t border-gray-800/60">
                    <h4 className="text-[9px] font-bold text-gray-500 uppercase tracking-widest mb-2">Quick Summary</h4>
                    <div className="grid grid-cols-2 gap-2 text-[10px]">
                        <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                            <span className="text-lg font-bold" style={{ color: scoreColor(overall_score) }}>{overall_score.toFixed(0)}</span>
                            <p className="text-[8px] text-gray-500">Overall</p>
                        </div>
                        <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                            <span className="text-lg font-bold text-white">{skin_age || '—'}</span>
                            <p className="text-[8px] text-gray-500">Skin Age</p>
                        </div>
                        <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                            <span className="text-lg font-bold text-yellow-400">{conditions.length}</span>
                            <p className="text-[8px] text-gray-500">Conditions</p>
                        </div>
                        <div className="bg-gray-800/60 rounded-lg p-2 text-center">
                            <span className="text-lg font-bold text-blue-400">{Object.keys(zone_scores).length}</span>
                            <p className="text-[8px] text-gray-500">Zones</p>
                        </div>
                    </div>
                </div>

                {/* Actions */}
                <div className="space-y-2 mt-auto">
                    <button onClick={printReport}
                        className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 rounded-lg font-medium text-sm transition-all">
                        🖨️ Print / Save PDF
                    </button>
                    <div className="flex gap-2">
                        <button onClick={() => setCurrentPage(Math.max(0, currentPage - 1))} disabled={currentPage === 0}
                            className="flex-1 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm disabled:opacity-30 transition-all">
                            ← Prev
                        </button>
                        <button onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))} disabled={currentPage === totalPages - 1}
                            className="flex-1 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm disabled:opacity-30 transition-all">
                            Next →
                        </button>
                    </div>
                </div>
            </div>

            {/* ── Right: Report Preview ── */}
            <div className="flex-1 flex items-center justify-center bg-gray-900 p-6 overflow-auto">
                <div ref={reportRef} className="bg-white rounded-lg shadow-2xl overflow-hidden" style={{ width: 595, height: 842, minWidth: 595, minHeight: 842 }}>
                    <div className="w-full h-full relative overflow-hidden">
                        {pages[currentPage]}
                    </div>
                </div>
            </div>
        </div>
    );
}
