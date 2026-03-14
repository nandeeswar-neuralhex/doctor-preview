import React, { useState, useMemo, useRef, useCallback, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Html } from '@react-three/drei';
import * as THREE from 'three';
import { computeFaceMeshTriangles } from './faceMeshTriangles';
import { SURGERY_CATEGORIES, applyAllDeformations, getAllProcedures } from './surgeryDeformations';

/**
 * SurgerySimulator — AURA-style 3D face surgery simulation.
 *
 * Interactive facial procedure simulator with per-feature sliders,
 * real-time 3D mesh deformation, surgery presets, and before/after.
 */

/* ── Surgery Presets ───────────────────────────────────────────── */

const PRESETS = [
    {
        id: 'rhinoplasty_refine',
        label: 'Rhinoplasty — Refine',
        icon: '👃',
        desc: 'Sharper bridge + narrower nostrils + upturned tip',
        values: { nose_sharpen: 0.7, nose_narrow: 0.5, nose_tip_up: 0.4 },
    },
    {
        id: 'rhinoplasty_natural',
        label: 'Rhinoplasty — Natural',
        icon: '👃',
        desc: 'Subtle bridge narrowing + hump removal',
        values: { nose_sharpen: 0.4, nose_bridge_reduce: 0.5, nose_narrow: 0.3 },
    },
    {
        id: 'v_line',
        label: 'V-Line Surgery',
        icon: '🗿',
        desc: 'Chin narrowing + jaw slimming + cheek reduction',
        values: { chin_sharpen: 0.7, jaw_slim: 0.6, cheek_reduce: 0.5 },
    },
    {
        id: 'full_lift',
        label: 'Full Face Lift',
        icon: '✨',
        desc: 'Brow lift + cheek lift + eye opening + lip lift',
        values: { brow_lift: 0.5, cheek_lift: 0.4, eye_open: 0.4, lip_lift: 0.3 },
    },
    {
        id: 'eye_rejuvenation',
        label: 'Eye Rejuvenation',
        icon: '👁️',
        desc: 'Open eyes + under-eye fix + brow lift',
        values: { eye_open: 0.5, under_eye_smooth: 0.6, brow_lift: 0.3, brow_inner_lift: 0.3 },
    },
    {
        id: 'lip_enhancement',
        label: 'Lip Enhancement',
        icon: '👄',
        desc: 'Fuller lips + corner lift + defined cupid\'s bow',
        values: { lip_augment: 0.6, lip_lift: 0.3, lip_corner_lift: 0.4 },
    },
    {
        id: 'jawline_sculpt',
        label: 'Jawline Sculpt',
        icon: '💪',
        desc: 'Defined jaw + chin augmentation + cheek contour',
        values: { jaw_define: 0.6, chin_augment: 0.4, cheek_contour: 0.5 },
    },
    {
        id: 'cat_eye',
        label: 'Cat Eye Look',
        icon: '😼',
        desc: 'Eye lift + brow arch + cheekbone projection',
        values: { cantho_lift: 0.6, brow_arch: 0.5, cheek_augment: 0.4 },
    },
];

/* ── 3D Deformed Face Mesh ─────────────────────────────────────── */

function DeformedFaceMesh({ landmarks3d, deformedLandmarks, faceTexture, showWireframe, ghostOriginal }) {
    const meshRef = useRef();
    const ghostRef = useRef();

    // Build geometry from landmarks
    const buildGeometry = useCallback((lm) => {
        if (!lm || lm.length < 468) return null;
        const geo = new THREE.BufferGeometry();
        const positions = new Float32Array(lm.length * 3);
        const uvCoords = new Float32Array(lm.length * 2);

        for (let i = 0; i < lm.length; i++) {
            const { x, y, z } = lm[i];
            positions[i * 3]     = (x - 0.5) * 2;
            positions[i * 3 + 1] = -(y - 0.5) * 2;
            positions[i * 3 + 2] = -z * 1.5;
            uvCoords[i * 2]     = x;
            uvCoords[i * 2 + 1] = 1 - y;
        }

        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('uv', new THREE.BufferAttribute(uvCoords, 2));

        const triangles = computeFaceMeshTriangles(lm);
        if (triangles.length > 0) {
            const indices = new Uint16Array(triangles.length * 3);
            triangles.forEach((tri, i) => {
                indices[i * 3]     = tri[0];
                indices[i * 3 + 1] = tri[1];
                indices[i * 3 + 2] = tri[2];
            });
            geo.setIndex(new THREE.BufferAttribute(indices, 1));
        }
        geo.computeVertexNormals();
        return geo;
    }, []);

    // Deformed mesh
    const deformedGeo = useMemo(
        () => buildGeometry(deformedLandmarks || landmarks3d),
        [deformedLandmarks, landmarks3d, buildGeometry]
    );

    // Original ghost mesh
    const originalGeo = useMemo(
        () => ghostOriginal ? buildGeometry(landmarks3d) : null,
        [landmarks3d, ghostOriginal, buildGeometry]
    );

    // Texture
    const texture = useMemo(() => {
        if (!faceTexture) return null;
        const tex = new THREE.TextureLoader().load(faceTexture);
        tex.flipY = false;
        tex.colorSpace = THREE.SRGBColorSpace;
        return tex;
    }, [faceTexture]);

    // Compute vertex displacement colors (green=no change, magenta=deformed)
    const displacementColors = useMemo(() => {
        if (!landmarks3d || !deformedLandmarks) return null;
        const colors = new Float32Array(landmarks3d.length * 3);
        let maxDisp = 0;

        const disps = landmarks3d.map((orig, i) => {
            const def = deformedLandmarks[i];
            const dx = def.x - orig.x;
            const dy = def.y - orig.y;
            const dz = def.z - orig.z;
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            maxDisp = Math.max(maxDisp, d);
            return d;
        });

        if (maxDisp < 0.0001) maxDisp = 1;

        for (let i = 0; i < landmarks3d.length; i++) {
            const t = Math.min(disps[i] / (maxDisp * 0.7), 1.0);
            // Green (unchanged) → Cyan → Magenta (max deformed)
            colors[i * 3]     = t * 0.9;               // R
            colors[i * 3 + 1] = (1 - t) * 0.8 + 0.1;  // G
            colors[i * 3 + 2] = t * 0.95;              // B
        }
        return colors;
    }, [landmarks3d, deformedLandmarks]);

    useEffect(() => {
        if (deformedGeo && displacementColors && showWireframe) {
            deformedGeo.setAttribute('color', new THREE.BufferAttribute(displacementColors, 3));
        }
    }, [deformedGeo, displacementColors, showWireframe]);

    if (!deformedGeo) return null;

    return (
        <group>
            {/* Ghost original (transparent wireframe) */}
            {ghostOriginal && originalGeo && (
                <mesh ref={ghostRef} geometry={originalGeo}>
                    <meshBasicMaterial wireframe color="#ffffff" transparent opacity={0.12} />
                </mesh>
            )}

            {/* Deformed mesh */}
            <mesh ref={meshRef} geometry={deformedGeo}>
                {showWireframe ? (
                    <meshStandardMaterial vertexColors side={THREE.DoubleSide} wireframe transparent opacity={0.85} />
                ) : texture ? (
                    <meshStandardMaterial map={texture} side={THREE.DoubleSide} />
                ) : (
                    <meshStandardMaterial color="#8b7d6b" side={THREE.DoubleSide} />
                )}
            </mesh>
        </group>
    );
}

/* ── Intensity Slider Component ────────────────────────────────── */

function ProcedureSlider({ procedure, value, onChange, color }) {
    return (
        <div className="group">
            <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] text-gray-300 group-hover:text-white transition-colors truncate flex-1 mr-2">
                    {procedure.label}
                </span>
                <span className="text-[9px] font-mono w-8 text-right" style={{ color: value > 0 ? color : '#6b7280' }}>
                    {(value * 100).toFixed(0)}%
                </span>
            </div>
            <div className="relative">
                <input
                    type="range"
                    min={0}
                    max={100}
                    value={value * 100}
                    onChange={e => onChange(Number(e.target.value) / 100)}
                    className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                    style={{
                        background: `linear-gradient(to right, ${color}44 0%, ${color} ${value * 100}%, #374151 ${value * 100}%, #374151 100%)`,
                        accentColor: color,
                    }}
                    title={procedure.description}
                />
            </div>
            <p className="text-[8px] text-gray-600 mt-0.5 truncate opacity-0 group-hover:opacity-100 transition-opacity">
                {procedure.description}
            </p>
        </div>
    );
}

/* ── Active Procedures Badge Bar ───────────────────────────────── */

function ActiveBadges({ deformations, onClear }) {
    const active = Object.entries(deformations).filter(([, v]) => v > 0);
    if (active.length === 0) return null;

    return (
        <div className="flex flex-wrap gap-1 p-2 bg-gray-900/60 rounded-lg border border-gray-700/40">
            {active.map(([procId, val]) => (
                <span key={procId}
                    className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-indigo-600/30 border border-indigo-500/40 text-[8px] text-indigo-300 cursor-pointer hover:bg-red-600/30 hover:border-red-500/40 hover:text-red-300 transition-all"
                    onClick={() => onClear(procId)}
                    title={`Click to remove ${procId}`}
                >
                    {procId.replace(/_/g, ' ')}
                    <span className="font-bold">{(val * 100).toFixed(0)}%</span>
                    <span className="ml-0.5">×</span>
                </span>
            ))}
        </div>
    );
}

/* ── Main SurgerySimulator Component ───────────────────────────── */

export default function SurgerySimulator({ result }) {
    const [deformations, setDeformations] = useState({});
    const [activeCategory, setActiveCategory] = useState('nose');
    const [showWireframe, setShowWireframe] = useState(false);
    const [showGhost, setShowGhost] = useState(false);
    const [compareMode, setCompareMode] = useState(false);
    const [showPresets, setShowPresets] = useState(false);
    const [splitPosition, setSplitPosition] = useState(50);
    const splitRef = useRef(null);
    const isDragging = useRef(false);

    if (!result?.landmarks_3d) {
        return (
            <div className="flex items-center justify-center h-full bg-black text-gray-500">
                <div className="text-center space-y-3">
                    <p className="text-6xl">🔬</p>
                    <p className="text-lg font-semibold text-gray-300">Surgery Simulator</p>
                    <p className="text-sm text-gray-500">Run a face analysis first to generate the 3D model</p>
                    <p className="text-[10px] text-gray-600">Guided Capture → Results → Surgery Sim</p>
                </div>
            </div>
        );
    }

    const { landmarks_3d, original_image } = result;
    const faceTexSrc = original_image ? `data:image/jpeg;base64,${original_image}` : null;

    // Compute deformed landmarks whenever deformations change
    const deformedLandmarks = useMemo(
        () => applyAllDeformations(landmarks_3d, deformations),
        [landmarks_3d, deformations]
    );

    const hasDeformations = Object.values(deformations).some(v => v > 0);
    const totalProcedures = Object.values(deformations).filter(v => v > 0).length;

    // Update single deformation
    const setIntensity = useCallback((procId, value) => {
        setDeformations(prev => ({ ...prev, [procId]: value }));
    }, []);

    // Clear single
    const clearProcedure = useCallback((procId) => {
        setDeformations(prev => {
            const next = { ...prev };
            delete next[procId];
            return next;
        });
    }, []);

    // Reset all
    const resetAll = useCallback(() => setDeformations({}), []);

    // Apply preset
    const applyPreset = useCallback((preset) => {
        setDeformations(prev => ({ ...prev, ...preset.values }));
        setShowPresets(false);
    }, []);

    // Split-view drag handler
    const handleSplitMouseDown = useCallback(() => { isDragging.current = true; }, []);
    useEffect(() => {
        const handleMove = (e) => {
            if (!isDragging.current || !splitRef.current) return;
            const rect = splitRef.current.getBoundingClientRect();
            const pct = ((e.clientX - rect.left) / rect.width) * 100;
            setSplitPosition(Math.max(10, Math.min(90, pct)));
        };
        const handleUp = () => { isDragging.current = false; };
        window.addEventListener('mousemove', handleMove);
        window.addEventListener('mouseup', handleUp);
        return () => {
            window.removeEventListener('mousemove', handleMove);
            window.removeEventListener('mouseup', handleUp);
        };
    }, []);

    return (
        <div className="flex h-full bg-black text-white select-none">
            {/* ═══════════════ LEFT PANEL: Controls ═══════════════ */}
            <div className="w-[280px] bg-gray-950/95 backdrop-blur border-r border-gray-800/50 flex flex-col shrink-0">
                {/* Header */}
                <div className="p-3 border-b border-gray-800/50">
                    <div className="flex items-center gap-2 mb-2">
                        <span className="text-lg">🔬</span>
                        <h2 className="text-sm font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                            Surgery Simulator
                        </h2>
                        {hasDeformations && (
                            <span className="ml-auto px-1.5 py-0.5 rounded-full bg-indigo-600/40 text-[9px] text-indigo-300 font-semibold">
                                {totalProcedures} active
                            </span>
                        )}
                    </div>

                    {/* Action buttons */}
                    <div className="flex gap-1.5">
                        <button
                            onClick={() => setShowPresets(!showPresets)}
                            className={`flex-1 py-1.5 rounded-lg text-[10px] font-semibold transition-all ${
                                showPresets ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                        >
                            ✨ Presets
                        </button>
                        <button
                            onClick={resetAll}
                            disabled={!hasDeformations}
                            className="flex-1 py-1.5 rounded-lg text-[10px] font-semibold bg-gray-800 text-gray-400 hover:bg-red-900/50 hover:text-red-300 transition-all disabled:opacity-30"
                        >
                            ↺ Reset All
                        </button>
                        <button
                            onClick={() => setCompareMode(!compareMode)}
                            disabled={!hasDeformations}
                            className={`flex-1 py-1.5 rounded-lg text-[10px] font-semibold transition-all disabled:opacity-30 ${
                                compareMode ? 'bg-green-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                        >
                            {compareMode ? '◑ Split' : '◑ Compare'}
                        </button>
                    </div>
                </div>

                {/* Presets dropdown */}
                {showPresets && (
                    <div className="p-2 border-b border-gray-800/50 bg-gray-900/60 max-h-[200px] overflow-y-auto">
                        <div className="grid grid-cols-1 gap-1.5">
                            {PRESETS.map(preset => (
                                <button
                                    key={preset.id}
                                    onClick={() => applyPreset(preset)}
                                    className="flex items-start gap-2 p-2 rounded-lg bg-gray-800/60 hover:bg-indigo-900/40 border border-gray-700/40 hover:border-indigo-500/40 transition-all text-left"
                                >
                                    <span className="text-base mt-0.5">{preset.icon}</span>
                                    <div className="flex-1 min-w-0">
                                        <p className="text-[10px] font-semibold text-gray-200">{preset.label}</p>
                                        <p className="text-[8px] text-gray-500 truncate">{preset.desc}</p>
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Active badges */}
                {hasDeformations && (
                    <div className="px-2 pt-2">
                        <ActiveBadges deformations={deformations} onClear={clearProcedure} />
                    </div>
                )}

                {/* Category tabs */}
                <div className="px-2 pt-3">
                    <div className="flex flex-wrap gap-1">
                        {Object.entries(SURGERY_CATEGORIES).map(([catId, cat]) => {
                            const activeSurgCount = Object.entries(cat.procedures).filter(
                                ([pid]) => deformations[pid] > 0
                            ).length;
                            return (
                                <button
                                    key={catId}
                                    onClick={() => setActiveCategory(catId)}
                                    className={`flex items-center gap-1 px-2 py-1 rounded-lg text-[10px] font-medium transition-all ${
                                        activeCategory === catId
                                            ? 'text-white shadow-lg'
                                            : 'bg-gray-800/60 text-gray-400 hover:bg-gray-700'
                                    }`}
                                    style={activeCategory === catId ? { background: cat.color + '33', borderColor: cat.color + '88', border: '1px solid' } : {}}
                                >
                                    <span className="text-xs">{cat.icon}</span>
                                    <span>{cat.label.split(' ')[0]}</span>
                                    {activeSurgCount > 0 && (
                                        <span className="w-3.5 h-3.5 rounded-full bg-indigo-500 text-[7px] font-bold flex items-center justify-center">
                                            {activeSurgCount}
                                        </span>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Procedure sliders */}
                <div className="flex-1 overflow-y-auto px-3 py-3 space-y-3">
                    {SURGERY_CATEGORIES[activeCategory] && (
                        <>
                            <div className="flex items-center gap-2 mb-1">
                                <span className="text-base">{SURGERY_CATEGORIES[activeCategory].icon}</span>
                                <h3 className="text-xs font-bold" style={{ color: SURGERY_CATEGORIES[activeCategory].color }}>
                                    {SURGERY_CATEGORIES[activeCategory].label}
                                </h3>
                            </div>
                            {Object.entries(SURGERY_CATEGORIES[activeCategory].procedures).map(([procId, proc]) => (
                                <ProcedureSlider
                                    key={procId}
                                    procedure={proc}
                                    value={deformations[procId] || 0}
                                    onChange={(val) => setIntensity(procId, val)}
                                    color={SURGERY_CATEGORIES[activeCategory].color}
                                />
                            ))}
                        </>
                    )}
                </div>

                {/* Bottom controls */}
                <div className="p-3 border-t border-gray-800/50 space-y-2">
                    <div className="flex gap-2">
                        <label className="flex items-center gap-1.5 text-[10px] text-gray-400 cursor-pointer">
                            <input type="checkbox" checked={showWireframe} onChange={e => setShowWireframe(e.target.checked)}
                                className="accent-indigo-500 w-3 h-3" />
                            Deformation Map
                        </label>
                        <label className="flex items-center gap-1.5 text-[10px] text-gray-400 cursor-pointer">
                            <input type="checkbox" checked={showGhost} onChange={e => setShowGhost(e.target.checked)}
                                className="accent-indigo-500 w-3 h-3" />
                            Ghost Original
                        </label>
                    </div>
                    <div className="text-[8px] text-gray-600 space-y-0.5">
                        <p>🖱 Drag to rotate  ·  🔍 Scroll to zoom  ·  ⇧+drag to pan</p>
                        <p>Deformations are real-time — adjust sliders to preview surgery results</p>
                    </div>
                </div>
            </div>

            {/* ═══════════════ CENTER: 3D Viewport ═══════════════ */}
            <div className="flex-1 relative" ref={splitRef}>
                {compareMode && hasDeformations ? (
                    /* ── Split View: Before | After ── */
                    <>
                        {/* After (deformed) — left side */}
                        <div className="absolute inset-0" style={{ clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }}>
                            <Canvas>
                                <PerspectiveCamera makeDefault position={[0, 0, 2.2]} fov={45} />
                                <ambientLight intensity={0.7} />
                                <directionalLight position={[2, 2, 3]} intensity={0.8} />
                                <directionalLight position={[-2, -1, 2]} intensity={0.3} />
                                <pointLight position={[0, 0, 3]} intensity={0.5} />
                                <DeformedFaceMesh
                                    landmarks3d={landmarks_3d}
                                    deformedLandmarks={deformedLandmarks}
                                    faceTexture={faceTexSrc}
                                    showWireframe={showWireframe}
                                    ghostOriginal={false}
                                />
                                <OrbitControls enableDamping dampingFactor={0.08} minDistance={1} maxDistance={5} />
                            </Canvas>
                        </div>

                        {/* Before (original) — right side */}
                        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${splitPosition}%)` }}>
                            <Canvas>
                                <PerspectiveCamera makeDefault position={[0, 0, 2.2]} fov={45} />
                                <ambientLight intensity={0.7} />
                                <directionalLight position={[2, 2, 3]} intensity={0.8} />
                                <directionalLight position={[-2, -1, 2]} intensity={0.3} />
                                <pointLight position={[0, 0, 3]} intensity={0.5} />
                                <DeformedFaceMesh
                                    landmarks3d={landmarks_3d}
                                    deformedLandmarks={landmarks_3d}
                                    faceTexture={faceTexSrc}
                                    showWireframe={false}
                                    ghostOriginal={false}
                                />
                                <OrbitControls enableDamping dampingFactor={0.08} minDistance={1} maxDistance={5} />
                            </Canvas>
                        </div>

                        {/* Split divider */}
                        <div
                            className="absolute top-0 bottom-0 w-1 cursor-col-resize z-30 group"
                            style={{ left: `${splitPosition}%`, transform: 'translateX(-50%)' }}
                            onMouseDown={handleSplitMouseDown}
                        >
                            <div className="absolute inset-y-0 -left-1 -right-1 bg-white/20 group-hover:bg-white/40 transition-colors" />
                            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 rounded-full bg-white/90 shadow-lg flex items-center justify-center">
                                <span className="text-black text-xs font-bold">⟷</span>
                            </div>
                        </div>

                        {/* Labels */}
                        <div className="absolute top-3 left-3 z-20 bg-green-600/80 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold">
                            AFTER
                        </div>
                        <div className="absolute top-3 right-3 z-20 bg-gray-600/80 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold">
                            BEFORE
                        </div>
                    </>
                ) : (
                    /* ── Normal View: Single Viewport ── */
                    <>
                        <Canvas>
                            <PerspectiveCamera makeDefault position={[0, 0, 2.2]} fov={45} />
                            <ambientLight intensity={0.7} />
                            <directionalLight position={[2, 2, 3]} intensity={0.8} />
                            <directionalLight position={[-2, -1, 2]} intensity={0.3} />
                            <pointLight position={[0, 0, 3]} intensity={0.5} />
                            <hemisphereLight args={['#b1e1ff', '#b97a20', 0.3]} />

                            <DeformedFaceMesh
                                landmarks3d={landmarks_3d}
                                deformedLandmarks={deformedLandmarks}
                                faceTexture={faceTexSrc}
                                showWireframe={showWireframe}
                                ghostOriginal={showGhost && hasDeformations}
                            />

                            <OrbitControls
                                enableDamping
                                dampingFactor={0.08}
                                autoRotate={!hasDeformations}
                                autoRotateSpeed={0.4}
                                minDistance={1}
                                maxDistance={5}
                            />
                        </Canvas>

                        {/* Status overlay */}
                        <div className="absolute top-3 left-3 flex items-center gap-2">
                            <div className="bg-black/60 backdrop-blur px-3 py-1.5 rounded-lg text-[10px] text-white font-semibold">
                                {hasDeformations ? (
                                    <span className="text-green-400">🔬 {totalProcedures} Procedure{totalProcedures > 1 ? 's' : ''} Applied</span>
                                ) : (
                                    <span className="text-gray-400">Adjust sliders to preview</span>
                                )}
                            </div>
                            {showWireframe && (
                                <div className="bg-purple-900/60 backdrop-blur px-2 py-1 rounded-lg text-[9px] text-purple-300">
                                    <span className="inline-block w-2 h-2 rounded-full bg-purple-400 mr-1" />
                                    Magenta = deformed area
                                </div>
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
}
