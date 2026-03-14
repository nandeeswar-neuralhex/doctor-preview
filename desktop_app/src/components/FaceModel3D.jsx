import React, { useMemo, useRef, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { computeFaceMeshTriangles } from './faceMeshTriangles';

/**
 * FaceModel3D — AURA-identical 3D face viewer.
 *
 * Matches AURA's exact UI: patient header, pill-style navigation
 * (Expressions · Measurements · Skin analysis · Volume · Vectors),
 * left vertical toolbar, skin analysis overlays (Wrinkles, Texture,
 * Brown Spots, Red Areas, Pores), regional scores face diagram,
 * severity filter, opacity slider, and timestamp footer.
 */

/* ── Severity level definitions ────────────────────────────────── */
const SEVERITY_LEVELS = [
    { level: 1, label: 'very mild', color: '#22c55e' },
    { level: 2, label: 'mild',      color: '#84cc16' },
    { level: 3, label: 'medium',    color: '#eab308' },
    { level: 4, label: 'severe',    color: '#f97316' },
    { level: 5, label: 'very severe', color: '#ef4444' },
];

/* ── Main navigation tabs (AURA-style) ─────────────────────────── */
const MAIN_TABS = [
    { id: 'expressions',  label: 'Expressions',   icon: '☺' },
    { id: 'measurements', label: 'Measurements',  icon: '⌨' },
    { id: 'skin',         label: 'Skin analysis', icon: '⇣⇣' },
    { id: 'volume',       label: 'Volume',        icon: '⟲' },
    { id: 'vectors',      label: 'Vectors',       icon: '⇢' },
];

const EXPRESSION_TABS = ['Neutral', 'Angry', 'Smile', 'Surprise', 'Kiss'];

const SKIN_TABS = [
    { id: 'wrinkles',    label: 'Wrinkles' },
    { id: 'texture',     label: 'Texture' },
    { id: 'brown_spots', label: 'Brown Spots' },
    { id: 'red_areas',   label: 'Red Areas' },
    { id: 'pores',       label: 'Pores' },
];

/* ── Left toolbar icons (AURA vertical bar) ────────────────────── */
const TOOLBAR_ICONS = [
    { id: 'undo',     icon: '↩',  label: 'Undo' },
    { id: 'zoom',     icon: '⌕',  label: 'Search' },
    { id: 'zoom_in',  icon: '⊕',  label: 'Zoom In' },
    { id: 'zoom_out', icon: '⊖',  label: 'Zoom Out' },
    { id: 'reset',    icon: '✋',  label: 'Reset / Pan' },
];

/* ── Auto-mask landmark regions (AURA: eyebrows, eyes, beard, lips) ── */
const MASK_REGIONS = {
    left_eyebrow:  { label: 'Left Eyebrow',  indices: [70,63,105,66,107,55,65,52,53,46,225,224,223,222,221], color: '#1a1a1a' },
    right_eyebrow: { label: 'Right Eyebrow', indices: [300,293,334,296,336,285,295,282,283,276,445,444,443,442,441], color: '#1a1a1a' },
    left_eye:      { label: 'Left Eye',      indices: [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,130,247,30,29,27,28,56,190,243,112,26,22,23,24,110,25], color: '#1a1a1a' },
    right_eye:     { label: 'Right Eye',     indices: [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,359,467,260,259,257,258,286,414,463,341,256,252,253,254,339,255], color: '#1a1a1a' },
    upper_lip:     { label: 'Upper Lip',     indices: [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,76,77,90,180,85,16,315,404,320,307,306,292], color: '#2a1a1a' },
    lower_lip:     { label: 'Lower Lip',     indices: [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,78,95,88,178,87,14,317,402,318,324,308,415], color: '#2a1a1a' },
    beard_zone:    { label: 'Beard / Chin',   indices: [152,175,199,200,18,313,421,396,428,377,400,378,379,365,397,288,361,323,454,356,389,251,284,332,297,338,10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,149,148,176,140,171,395,369], color: '#1a1a1a' },
    mustache:      { label: 'Mustache',      indices: [164,167,165,92,186,57,43,106,182,83,18,313,406,335,273,287,410,322,391,393], color: '#1a1a1a' },
};

const DEFAULT_MASK = { left_eyebrow: true, right_eyebrow: true, left_eye: true, right_eye: true, upper_lip: false, lower_lip: false, beard_zone: true, mustache: true };

/* ── Deterministic pseudo-random ───────────────────────────────── */
function seededRng(seed) {
    let s = seed;
    return () => { s = (s * 16807) % 2147483647; return s / 2147483647; };
}

/* ── 3D Face Mesh ──────────────────────────────────────────────── */
function FaceMesh3D({ landmarks3d, faceTexture, viewMode, overlayOpacity, skinTab, zoneScores, severityFilter, maskRegions, showMask }) {
    const meshRef = useRef();

    /* Pre-compute masked landmark set for particle filtering */
    const maskedLandmarks = useMemo(() => {
        if (!showMask || !maskRegions) return null;
        const s = new Set();
        Object.entries(MASK_REGIONS).forEach(([key, region]) => {
            if (maskRegions[key]) region.indices.forEach(i => s.add(i));
        });
        return s.size > 0 ? s : null;
    }, [showMask, maskRegions]);

    const geometry = useMemo(() => {
        if (!landmarks3d || landmarks3d.length < 468) return null;
        const geo = new THREE.BufferGeometry();
        const pos = new Float32Array(landmarks3d.length * 3);
        const uv = new Float32Array(landmarks3d.length * 2);

        for (let i = 0; i < landmarks3d.length; i++) {
            const { x, y, z } = landmarks3d[i];
            pos[i * 3]     = (x - 0.5) * 2;
            pos[i * 3 + 1] = -(y - 0.5) * 2;
            pos[i * 3 + 2] = -z * 1.5;
            uv[i * 2]      = x;
            uv[i * 2 + 1]  = 1 - y;
        }
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));

        const tris = computeFaceMeshTriangles(landmarks3d);
        if (tris.length > 0) {
            const idx = new Uint16Array(tris.length * 3);
            tris.forEach((t, i) => { idx[i * 3] = t[0]; idx[i * 3 + 1] = t[1]; idx[i * 3 + 2] = t[2]; });
            geo.setIndex(new THREE.BufferAttribute(idx, 1));
        }
        geo.computeVertexNormals();
        return geo;
    }, [landmarks3d]);

    const texture = useMemo(() => {
        if (!faceTexture) return null;
        const t = new THREE.TextureLoader().load(faceTexture);
        t.flipY = false;
        t.colorSpace = THREE.SRGBColorSpace;
        return t;
    }, [faceTexture]);

    /* ── Skin-analysis particle dots (like AURA) ── */
    const particleGeo = useMemo(() => {
        if (viewMode !== 'skin' || !landmarks3d) return null;

        const typeColors = {
            wrinkles:    { base: [0.85, 0.25, 0.2],  accent: [1.0, 0.5, 0.15] },
            texture:     { base: [0.7, 0.15, 0.55],  accent: [0.9, 0.25, 0.7] },
            brown_spots: { base: [0.6, 0.38, 0.08],  accent: [0.8, 0.5, 0.12] },
            red_areas:   { base: [1.0, 0.1, 0.1],    accent: [1.0, 0.25, 0.15] },
            pores:       { base: [0.2, 0.7, 0.25],   accent: [0.9, 0.82, 0.08] },
        };
        const col = typeColors[skinTab] || typeColors.pores;
        const rand = seededRng(skinTab ? skinTab.charCodeAt(0) * 137 + skinTab.length : 42);

        const zoneRanges = {
            forehead: { yMin: 0, yMax: 0.28 },
            eyes:     { yMin: 0.28, yMax: 0.42 },
            nose:     { yMin: 0.35, yMax: 0.55 },
            cheeks:   { yMin: 0.42, yMax: 0.68 },
            mouth:    { yMin: 0.62, yMax: 0.78 },
            chin:     { yMin: 0.78, yMax: 1.0 },
        };

        const pts = [];
        for (let i = 0; i < 700; i++) {
            const lmIdx = Math.floor(rand() * landmarks3d.length);
            const lm = landmarks3d[lmIdx];
            /* Skip particles in masked regions — "No scoring" */
            if (maskedLandmarks && maskedLandmarks.has(lmIdx)) continue;
            let sev = 2;
            for (const [zone, range] of Object.entries(zoneRanges)) {
                if (lm.y >= range.yMin && lm.y <= range.yMax) {
                    const zk = zone === 'eyes' ? 'left_eye' : zone === 'cheeks' ? 'left_cheek' : zone;
                    const sc = zoneScores?.[zk]?.overall || zoneScores?.[zone]?.overall || 70;
                    sev = sc >= 85 ? 1 : sc >= 70 ? 2 : sc >= 50 ? 3 : sc >= 30 ? 4 : 5;
                    break;
                }
            }
            if (!severityFilter[sev]) continue;
            const jx = (rand() - 0.5) * 0.035;
            const jy = (rand() - 0.5) * 0.035;
            const m = rand();
            pts.push({
                x: (lm.x + jx - 0.5) * 2,
                y: -(lm.y + jy - 0.5) * 2,
                z: -lm.z * 1.5 + 0.02,
                r: col.base[0] * (1 - m) + col.accent[0] * m,
                g: col.base[1] * (1 - m) + col.accent[1] * m,
                b: col.base[2] * (1 - m) + col.accent[2] * m,
            });
        }

        if (pts.length === 0) return null;
        const geo = new THREE.BufferGeometry();
        const p = new Float32Array(pts.length * 3);
        const c = new Float32Array(pts.length * 3);
        pts.forEach((pt, i) => {
            p[i * 3] = pt.x; p[i * 3 + 1] = pt.y; p[i * 3 + 2] = pt.z;
            c[i * 3] = pt.r; c[i * 3 + 1] = pt.g; c[i * 3 + 2] = pt.b;
        });
        geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
        return geo;
    }, [viewMode, skinTab, landmarks3d, zoneScores, severityFilter, maskedLandmarks]);

    if (!geometry) return null;

    return (
        <group>
            <mesh ref={meshRef} geometry={geometry}>
                {viewMode === 'skin' && overlayOpacity < 1 ? (
                    <meshStandardMaterial map={texture} side={THREE.DoubleSide} transparent opacity={1 - overlayOpacity * 0.35} />
                ) : texture ? (
                    <meshStandardMaterial map={texture} side={THREE.DoubleSide} />
                ) : (
                    <meshStandardMaterial color="#8b7d6b" side={THREE.DoubleSide} />
                )}
            </mesh>
            {particleGeo && viewMode === 'skin' && (
                <points geometry={particleGeo}>
                    <pointsMaterial vertexColors size={0.018} transparent opacity={overlayOpacity} sizeAttenuation depthWrite={false} />
                </points>
            )}
            {/* Dark mask overlay patches */}
            {showMask && landmarks3d && <MaskOverlay landmarks3d={landmarks3d} maskRegions={maskRegions} />}
        </group>
    );
}

/* ── 3D Mask Overlay — solid dark filled patches (AURA-style) ── */
function MaskOverlay({ landmarks3d, maskRegions }) {
    const { solidGeo, pointGeo } = useMemo(() => {
        if (!landmarks3d || !maskRegions) return { solidGeo: null, pointGeo: null };
        const activeSet = new Set();
        Object.entries(MASK_REGIONS).forEach(([key, region]) => {
            if (maskRegions[key]) region.indices.forEach(i => activeSet.add(i));
        });
        if (activeSet.size === 0) return { solidGeo: null, pointGeo: null };

        /* Build full-face triangulation, then keep only triangles where ALL 3 verts are masked */
        const tris = computeFaceMeshTriangles(landmarks3d);
        const maskedTris = tris.filter(t => activeSet.has(t[0]) && activeSet.has(t[1]) && activeSet.has(t[2]));

        /* Re-index: map global indices → local dense buffer */
        const usedSet = new Set();
        maskedTris.forEach(t => { usedSet.add(t[0]); usedSet.add(t[1]); usedSet.add(t[2]); });
        const usedArr = [...usedSet];
        const g2l = new Map();
        usedArr.forEach((g, i) => g2l.set(g, i));

        const pos = new Float32Array(usedArr.length * 3);
        usedArr.forEach((gi, i) => {
            const lm = landmarks3d[gi];
            pos[i * 3]     = (lm.x - 0.5) * 2;
            pos[i * 3 + 1] = -(lm.y - 0.5) * 2;
            pos[i * 3 + 2] = -lm.z * 1.5 + 0.015;  // slight offset forward
        });

        const idx = new Uint16Array(maskedTris.length * 3);
        maskedTris.forEach((t, i) => {
            idx[i * 3]     = g2l.get(t[0]);
            idx[i * 3 + 1] = g2l.get(t[1]);
            idx[i * 3 + 2] = g2l.get(t[2]);
        });

        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setIndex(new THREE.BufferAttribute(idx, 1));
        geo.computeVertexNormals();

        /* Also generate border dots for a softer edge (AURA look) */
        const allIdx2 = [...activeSet];
        const dotPos = new Float32Array(allIdx2.length * 3);
        allIdx2.forEach((gi, i) => {
            const lm = landmarks3d[gi];
            if (!lm) return;
            dotPos[i * 3]     = (lm.x - 0.5) * 2;
            dotPos[i * 3 + 1] = -(lm.y - 0.5) * 2;
            dotPos[i * 3 + 2] = -lm.z * 1.5 + 0.02;
        });
        const pGeo = new THREE.BufferGeometry();
        pGeo.setAttribute('position', new THREE.BufferAttribute(dotPos, 3));

        return { solidGeo: geo, pointGeo: pGeo };
    }, [landmarks3d, maskRegions]);

    if (!solidGeo) return null;
    return (
        <group>
            <mesh geometry={solidGeo}>
                <meshStandardMaterial color="#0a0a0a" side={THREE.DoubleSide} transparent opacity={0.72} depthWrite={false} />
            </mesh>
            {pointGeo && (
                <points geometry={pointGeo}>
                    <pointsMaterial color="#080808" size={0.04} transparent opacity={0.85} sizeAttenuation depthWrite={false} />
                </points>
            )}
        </group>
    );
}

/* ── Regional Scores Face Diagram ──────────────────────────────── */
function RegionalScoresDiagram({ zoneScores }) {
    const toAura = (s) => (5 - Math.max(0, Math.min(100, s || 70)) / 25).toFixed(1);
    const zColor = (s) => (s || 70) >= 80 ? '#22c55e' : (s || 70) >= 60 ? '#a3e635' : (s || 70) >= 40 ? '#eab308' : '#f97316';

    const zones = [
        { id: 'forehead',    x: 50, y: 16, score: zoneScores?.forehead?.overall },
        { id: 'left_cheek',  x: 20, y: 55, score: zoneScores?.left_cheek?.overall },
        { id: 'nose',        x: 50, y: 46, score: zoneScores?.nose?.overall },
        { id: 'right_cheek', x: 80, y: 55, score: zoneScores?.right_cheek?.overall },
    ];

    return (
        <div className="bg-[#1e1e1e]/90 backdrop-blur-2xl rounded-2xl border border-gray-700/40 shadow-2xl overflow-hidden" style={{ width: 190 }}>
            <div className="flex items-center justify-between px-3 pt-2.5 pb-1">
                <h4 className="text-[11px] font-semibold text-white tracking-wide">Regional Scores</h4>
                <button className="text-gray-500 hover:text-white text-[10px]">⬜</button>
            </div>
            <div className="relative mx-auto" style={{ width: 150, height: 180 }}>
                <svg viewBox="0 0 100 120" className="w-full h-full absolute inset-0 opacity-20">
                    <ellipse cx="50" cy="55" rx="38" ry="50" fill="#555" />
                    <ellipse cx="35" cy="42" rx="8" ry="5" fill="#333" />
                    <ellipse cx="65" cy="42" rx="8" ry="5" fill="#333" />
                </svg>
                {zones.map(z => (
                    <div key={z.id} className="absolute flex flex-col items-center" style={{ left: `${z.x}%`, top: `${z.y}%`, transform: 'translate(-50%,-50%)' }}>
                        <div className="w-11 h-9 rounded-lg" style={{ background: zColor(z.score), opacity: 0.65 }} />
                        <span className="text-[10px] font-bold text-white mt-0.5 drop-shadow">{toAura(z.score)}</span>
                    </div>
                ))}
                <div className="absolute bottom-1 left-1/2 -translate-x-1/2 text-center">
                    <span className="text-[9px] text-yellow-500 drop-shadow">⚠</span>
                    <p className="text-[7px] text-gray-400 font-semibold leading-tight">Beard<br />Detected</p>
                </div>
            </div>
            <div className="flex items-center gap-0.5 px-3 pb-1.5">
                <span className="text-[6px] text-gray-500 mr-0.5">mild</span>
                {SEVERITY_LEVELS.map(s => (
                    <div key={s.level} className="flex-1 h-3 rounded-sm flex items-center justify-center text-[7px] font-bold text-white" style={{ background: s.color }}>{s.level}</div>
                ))}
                <span className="text-[6px] text-red-400 ml-0.5">severe</span>
            </div>
            <div className="flex justify-center gap-1 pb-2">
                <div className="w-1 h-1 rounded-full bg-white" /><div className="w-1 h-1 rounded-full bg-gray-600" /><div className="w-1 h-1 rounded-full bg-gray-600" />
            </div>
        </div>
    );
}

/* ── Severity Filter Panel ─────────────────────────────────────── */
function SeverityFilterPanel({ filter, onToggle, distribution }) {
    return (
        <div className="bg-[#1e1e1e]/90 backdrop-blur-2xl rounded-2xl border border-gray-700/40 shadow-2xl overflow-hidden" style={{ width: 190 }}>
            <div className="flex items-center justify-between px-3 pt-2.5 pb-1">
                <h4 className="text-[11px] font-semibold text-white tracking-wide">Severity Filter</h4>
                <button className="text-gray-500 hover:text-white text-[10px]">⬜</button>
            </div>
            <div className="px-2 pb-2 space-y-1">
                {SEVERITY_LEVELS.map(sev => {
                    const pct = distribution[sev.level] || 0;
                    const on = filter[sev.level];
                    return (
                        <button key={sev.level} onClick={() => onToggle(sev.level)}
                            className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-xl transition-all ${on ? '' : 'opacity-30'}`}
                            style={on ? { background: sev.color + '22' } : {}}>
                            <div className="w-5 h-5 rounded-lg flex items-center justify-center text-[9px] font-bold text-white" style={{ background: on ? sev.color : '#374151' }}>
                                {on ? '✓' : '●'}
                            </div>
                            <span className="flex-1 text-left text-[10px] font-semibold" style={{ color: on ? sev.color : '#4b5563' }}>
                                {sev.level} <span className="font-normal opacity-80">({sev.label})</span>
                            </span>
                            <span className="text-[11px] font-bold" style={{ color: on ? sev.color : '#4b5563' }}>{pct}%</span>
                        </button>
                    );
                })}
            </div>
            <div className="flex justify-center gap-1 pb-2">
                <div className="w-1 h-1 rounded-full bg-gray-600" /><div className="w-1 h-1 rounded-full bg-white" /><div className="w-1 h-1 rounded-full bg-gray-600" />
            </div>
        </div>
    );
}

/* ═══════════════════════════════════════════════════════════════════
   MAIN EXPORT
   ═══════════════════════════════════════════════════════════════════ */
export default function FaceModel3D({ result, compareResult, patientInfo }) {
    const [mainTab, setMainTab] = useState('expressions');
    const [expressionTab, setExpressionTab] = useState('Neutral');
    const [skinTab, setSkinTab] = useState('pores');
    const [overlayOpacity, setOverlayOpacity] = useState(0.7);
    const [activeTool, setActiveTool] = useState('rotate');
    const [infoPanel, setInfoPanel] = useState('regional');
    const [severityFilter, setSeverityFilter] = useState({ 1: true, 2: true, 3: true, 4: true, 5: true });
    const [maskMode, setMaskMode] = useState(false);
    const [maskRegions, setMaskRegions] = useState({ ...DEFAULT_MASK });
    const [maskBrush, setMaskBrush] = useState('remove'); // 'remove' | 'add'
    const [brushSize, setBrushSize] = useState(50);
    const [darkAreasEnabled, setDarkAreasEnabled] = useState(true);
    const controlsRef = useRef();

    const toggleSeverity = useCallback((lv) => setSeverityFilter(p => ({ ...p, [lv]: !p[lv] })), []);

    /* ── No data fallback ── */
    if (!result?.landmarks_3d) {
        return (
            <div className="flex items-center justify-center h-full bg-[#2a2a2a] text-gray-500">
                <div className="text-center space-y-3">
                    <p className="text-6xl">🧊</p>
                    <p className="text-lg font-semibold text-gray-300">Instant 3D capture of face and neck</p>
                    <p className="text-xs text-gray-500">Run Guided Capture → Results first</p>
                </div>
            </div>
        );
    }

    const { landmarks_3d, original_image, zone_scores = {}, skin_age } = result;
    const faceTexSrc = original_image ? `data:image/jpeg;base64,${original_image}` : null;
    const pName = patientInfo?.name || 'Patient';
    const pAge = patientInfo?.age || skin_age || '';
    const now = new Date();
    const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' });
    const timeStr = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: true }).toUpperCase();

    /* severity distribution */
    const sevDist = useMemo(() => {
        const ss = Object.values(zone_scores).map(z => z?.overall || 70);
        if (!ss.length) return { 1: 59, 2: 12, 3: 10, 4: 6, 5: 13 };
        const a = ss.reduce((x, y) => x + y, 0) / ss.length;
        return { 1: Math.round(a * 0.6), 2: Math.round((100 - a) * 0.15), 3: Math.round((100 - a) * 0.12), 4: Math.round((100 - a) * 0.08), 5: Math.round((100 - a) * 0.15) };
    }, [zone_scores]);

    const handleTool = useCallback((id) => {
        setActiveTool(id);
        if (id === 'reset' && controlsRef.current) controlsRef.current.reset();
    }, []);

    return (
        <div className="relative flex flex-col h-full bg-[#2a2a2a] text-white overflow-hidden select-none">

            {/* ═══ TOP: Patient header + nav pills ═══ */}
            <div className="relative z-20 flex flex-col items-center pt-3 pb-1 pointer-events-auto">
                {/* Back arrow (AURA top-left) */}
                <button className="absolute left-4 top-3 text-gray-400 hover:text-white text-lg transition-colors z-30">‹</button>
                {/* Patient name */}
                <p className="text-[15px] tracking-wide text-gray-300/90 mb-2.5" style={{ fontFamily: 'Georgia, "Times New Roman", serif', fontStyle: 'italic', letterSpacing: '0.5px' }}>
                    {pName} ({pAge})
                </p>

                {/* Main pill nav */}
                <div className="flex items-center gap-0.5 bg-[#3a3a3a]/70 backdrop-blur-xl rounded-full px-1 py-0.5 border border-gray-600/30">
                    {MAIN_TABS.map(tab => (
                        <button key={tab.id} onClick={() => setMainTab(tab.id)}
                            className={`px-3.5 py-1.5 rounded-full text-[11px] font-medium transition-all flex items-center gap-1.5 whitespace-nowrap ${
                                mainTab === tab.id ? 'bg-[#555]/80 text-white shadow-sm' : 'text-gray-400 hover:text-white'
                            }`}>
                            <span className="text-[10px]">{tab.icon}</span>{tab.label}
                        </button>
                    ))}
                </div>

                {/* Sub-tabs: Expressions */}
                {mainTab === 'expressions' && (
                    <div className="flex gap-0.5 mt-2 bg-[#3a3a3a]/50 rounded-full px-1 py-0.5">
                        {EXPRESSION_TABS.map(exp => (
                            <button key={exp} onClick={() => setExpressionTab(exp)}
                                className={`px-3 py-1 rounded-full text-[10px] font-medium transition-all ${
                                    expressionTab === exp ? 'bg-[#555]/70 text-white' : 'text-gray-500 hover:text-gray-300'
                                }`}>{exp}</button>
                        ))}
                    </div>
                )}

                {/* Sub-tabs: Skin analysis */}
                {mainTab === 'skin' && !maskMode && (
                    <div className="flex gap-0.5 mt-2 bg-[#3a3a3a]/50 rounded-full px-1 py-0.5">
                        {SKIN_TABS.map(tab => (
                            <button key={tab.id} onClick={() => setSkinTab(tab.id)}
                                className={`px-3 py-1 rounded-full text-[10px] font-medium transition-all ${
                                    skinTab === tab.id ? 'bg-[#555]/70 text-white' : 'text-gray-500 hover:text-gray-300'
                                }`}>{tab.label}</button>
                        ))}
                        <button onClick={() => setMaskMode(true)}
                            className="px-3 py-1 rounded-full text-[10px] font-medium text-orange-400 hover:text-orange-300 hover:bg-orange-900/20 transition-all ml-1">
                            🎭 Mask
                        </button>
                    </div>
                )}
            </div>

            {/* ═══ CENTER: Left tools + 3D viewport + info panels ═══ */}
            <div className="flex-1 flex relative min-h-0">

                {/* Left vertical toolbar */}
                <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 flex flex-col items-center gap-1.5">
                    {TOOLBAR_ICONS.map(t => (
                        <button key={t.id} onClick={() => handleTool(t.id)} title={t.label}
                            className={`w-7 h-7 rounded-full flex items-center justify-center text-[13px] transition-all ${
                                activeTool === t.id ? 'bg-gray-600/50 text-white' : 'text-gray-500 hover:text-white hover:bg-gray-700/30'
                            }`}>{t.icon}</button>
                    ))}
                    <div className="flex flex-col gap-1 mt-2">
                        {[0, 1, 2, 3, 4].map(i => <div key={i} className={`w-[5px] h-[5px] rounded-full ${i === 0 ? 'bg-white' : 'bg-gray-600'}`} />)}
                    </div>
                </div>

                {/* 3D Canvas */}
                <div className="flex-1">
                    <Canvas>
                        <PerspectiveCamera makeDefault position={[0, 0, 2.4]} fov={42} />
                        <ambientLight intensity={0.6} />
                        <directionalLight position={[2.5, 2, 3]} intensity={0.9} color="#fff5e6" />
                        <directionalLight position={[-2, -1, 2]} intensity={0.3} color="#e0e0ff" />
                        <pointLight position={[0, 0, 4]} intensity={0.35} />
                        <hemisphereLight args={['#c4deff', '#614a2a', 0.15]} />

                        <FaceMesh3D
                            landmarks3d={landmarks_3d}
                            faceTexture={faceTexSrc}
                            viewMode={mainTab}
                            overlayOpacity={overlayOpacity}
                            skinTab={skinTab}
                            zoneScores={zone_scores}
                            severityFilter={severityFilter}
                            maskRegions={maskRegions}
                            showMask={maskMode && darkAreasEnabled}
                        />

                        <OrbitControls ref={controlsRef} enableDamping dampingFactor={0.06}
                            autoRotate={mainTab === 'expressions'} autoRotateSpeed={0.25}
                            minDistance={1} maxDistance={5} />
                    </Canvas>
                </div>

                {/* Bottom-left info panels (skin mode only) */}
                {mainTab === 'skin' && (
                    <div className="absolute bottom-14 left-14 z-20">
                        <div className="flex items-center gap-1 mb-1.5">
                            <button onClick={() => setInfoPanel('regional')} className={`text-lg ${infoPanel !== 'regional' ? 'text-gray-600 hover:text-gray-400' : 'text-transparent pointer-events-none'}`}>‹</button>
                            <div className="flex-1" />
                            <button onClick={() => setInfoPanel('severity')} className={`text-lg ${infoPanel !== 'severity' ? 'text-gray-600 hover:text-gray-400' : 'text-transparent pointer-events-none'}`}>›</button>
                        </div>
                        {infoPanel === 'regional'
                            ? <RegionalScoresDiagram zoneScores={zone_scores} />
                            : <SeverityFilterPanel filter={severityFilter} onToggle={toggleSeverity} distribution={sevDist} />
                        }
                    </div>
                )}

                {/* Opacity slider — below panels, left-aligned (AURA style) */}
                {mainTab === 'skin' && !maskMode && (
                    <div className="absolute bottom-4 left-14 z-10 flex items-center gap-2 bg-[#1e1e1e]/70 backdrop-blur-xl rounded-full px-4 py-1.5 border border-gray-700/30" style={{ marginTop: 4, width: 190 }}>
                        <span className="text-[10px] text-gray-400 font-medium">Opacity</span>
                        <input type="range" min="0" max="1" step="0.05" value={overlayOpacity}
                            onChange={e => setOverlayOpacity(+e.target.value)}
                            className="flex-1 h-1 accent-gray-400 cursor-pointer" />
                    </div>
                )}

                {/* ═══ Mask Mode: Top toolbar (Remove/Add + rotation presets) ═══ */}
                {maskMode && (
                    <div className="absolute top-2 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2">
                        <div className="flex items-center gap-0.5 bg-[#3a3a3a]/80 backdrop-blur-xl rounded-full px-1 py-0.5 border border-gray-600/30">
                            <button onClick={() => setMaskBrush('remove')}
                                className={`px-3 py-1.5 rounded-full text-[10px] font-medium flex items-center gap-1.5 transition-all ${
                                    maskBrush === 'remove' ? 'bg-[#555]/80 text-white' : 'text-gray-400 hover:text-white'
                                }`}>
                                <span>😐</span> Remove
                            </button>
                            <button onClick={() => setMaskBrush('add')}
                                className={`px-3 py-1.5 rounded-full text-[10px] font-medium flex items-center gap-1.5 transition-all ${
                                    maskBrush === 'add' ? 'bg-[#555]/80 text-white' : 'text-gray-400 hover:text-white'
                                }`}>
                                <span>✏️</span> Add
                            </button>
                        </div>
                        <div className="flex items-center gap-1 bg-[#3a3a3a]/60 rounded-full px-2 py-1 border border-gray-600/20">
                            <button className="w-6 h-6 rounded-full text-[12px] text-gray-400 hover:text-white transition-all hover:bg-gray-600/40" title="Front view"
                                onClick={() => { if (controlsRef.current) { controlsRef.current.reset(); } }}>👤</button>
                            <button className="w-6 h-6 rounded-full text-[12px] text-gray-400 hover:text-white transition-all hover:bg-gray-600/40" title="Left profile">👤</button>
                            <button className="w-6 h-6 rounded-full text-[12px] text-gray-400 hover:text-white transition-all hover:bg-gray-600/40 scale-x-[-1]" title="Right profile">👤</button>
                        </div>
                    </div>
                )}

                {/* ═══ Mask Mode: Right Edit Mask Panel ═══ */}
                {maskMode && (
                    <div className="absolute top-2 right-3 z-30 w-[180px] bg-[#1e1e1e]/95 backdrop-blur-2xl rounded-2xl border border-gray-700/40 shadow-2xl overflow-hidden">
                        <div className="px-3 pt-3 pb-2">
                            <h4 className="text-[11px] font-semibold text-white mb-0.5">Edit Mask</h4>
                            <p className="text-[8px] text-gray-500">Brush to add or remove mask areas</p>
                        </div>

                        {/* Dark Areas toggle */}
                        <div className="px-3 pb-2">
                            <label className="flex items-center justify-between cursor-pointer">
                                <span className="text-[10px] text-gray-300">Dark Areas</span>
                                <div className={`w-8 h-4 rounded-full relative transition-all ${darkAreasEnabled ? 'bg-indigo-600' : 'bg-gray-600'}`}
                                    onClick={() => setDarkAreasEnabled(!darkAreasEnabled)}>
                                    <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${darkAreasEnabled ? 'left-4' : 'left-0.5'}`} />
                                </div>
                            </label>
                            <p className="text-[8px] text-gray-600 mt-1">No scoring in masked areas</p>
                        </div>

                        {/* Region toggles */}
                        <div className="px-3 pb-2 space-y-1">
                            <p className="text-[8px] text-gray-500 uppercase tracking-wider font-bold mb-1">Mask Regions</p>
                            {Object.entries(MASK_REGIONS).map(([key, region]) => (
                                <label key={key} className="flex items-center gap-2 cursor-pointer group">
                                    <div className={`w-4 h-4 rounded border-2 flex items-center justify-center text-[8px] transition-all ${
                                        maskRegions[key] ? 'bg-indigo-600 border-indigo-500 text-white' : 'border-gray-600 text-transparent group-hover:border-gray-400'
                                    }`}
                                        onClick={() => setMaskRegions(p => ({ ...p, [key]: !p[key] }))}>
                                        ✓
                                    </div>
                                    <span className={`text-[10px] transition-all ${maskRegions[key] ? 'text-gray-200' : 'text-gray-500'}`}>
                                        {region.label}
                                    </span>
                                </label>
                            ))}
                        </div>

                        {/* Brush size */}
                        <div className="px-3 pb-2">
                            <div className="flex items-center justify-between mb-1">
                                <span className="text-[9px] text-gray-500">Brush Size</span>
                                <span className="text-[9px] text-gray-400 font-mono">{brushSize}</span>
                            </div>
                            <input type="range" min="10" max="100" value={brushSize}
                                onChange={e => setBrushSize(+e.target.value)}
                                className="w-full h-1 accent-gray-400 cursor-pointer" />
                        </div>

                        {/* Actions */}
                        <div className="px-3 pb-3 flex gap-2">
                            <button onClick={() => setMaskRegions({ ...DEFAULT_MASK })}
                                className="flex-1 py-1.5 rounded-lg text-[9px] font-medium bg-gray-700/60 text-gray-300 hover:bg-gray-600 transition-all">
                                Reset
                            </button>
                            <button onClick={() => setMaskMode(false)}
                                className="flex-1 py-1.5 rounded-lg text-[9px] font-medium bg-indigo-600 text-white hover:bg-indigo-500 transition-all">
                                Done
                            </button>
                        </div>
                    </div>
                )}

                {/* Brush cursor indicator (mask mode) */}
                {maskMode && (
                    <div className="absolute inset-0 z-10 pointer-events-none flex items-center justify-center">
                        <div className="rounded-full border-2 border-white/40" style={{ width: brushSize * 0.8, height: brushSize * 0.8 }} />
                    </div>
                )}
            </div>

            {/* ═══ BOTTOM: Timestamp bar ═══ */}
            <div className="relative z-20 flex justify-center pb-3 pt-1">
                <div className="bg-[#4a4a3a]/60 backdrop-blur-md rounded-full px-5 py-1.5 flex items-center gap-2 border border-gray-600/20">
                    <span className="text-[10px] text-gray-400">☰</span>
                    <span className="text-[10px] text-gray-300 font-medium tracking-wide">{dateStr} - {timeStr}</span>
                </div>
            </div>
        </div>
    );
}
