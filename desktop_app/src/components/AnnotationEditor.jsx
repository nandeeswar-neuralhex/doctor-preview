import React, { useRef, useState, useEffect, useCallback } from 'react';

/**
 * AnnotationEditor — Canvas-based drawing and annotation tool on face analysis images.
 *
 * Features:
 *  - Freehand draw (pen / marker)
 *  - Shapes (rectangle, circle, arrow, line)
 *  - Text annotations
 *  - Eraser
 *  - Color & stroke controls
 *  - Undo / redo stack
 *  - Save annotations to server
 *  - Export annotated image as PNG
 */

const TOOLS = [
    { id: 'pen',       icon: '✏️', label: 'Pen' },
    { id: 'marker',    icon: '🖍️', label: 'Marker' },
    { id: 'line',      icon: '📏', label: 'Line' },
    { id: 'arrow',     icon: '➡️', label: 'Arrow' },
    { id: 'rect',      icon: '⬜', label: 'Rectangle' },
    { id: 'circle',    icon: '⭕', label: 'Circle' },
    { id: 'text',      icon: '🔤', label: 'Text' },
    { id: 'eraser',    icon: '🧹', label: 'Eraser' },
];

const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#ffffff'];
const STROKE_WIDTHS = [1, 2, 4, 6, 10];

export default function AnnotationEditor({ result, serverUrl, sessionId }) {
    const canvasRef = useRef(null);
    const overlayRef = useRef(null);
    const containerRef = useRef(null);

    const [tool, setTool] = useState('pen');
    const [color, setColor] = useState('#ef4444');
    const [strokeWidth, setStrokeWidth] = useState(2);
    const [drawing, setDrawing] = useState(false);
    const [startPos, setStartPos] = useState(null);
    const [annotations, setAnnotations] = useState([]);   // committed strokes / shapes
    const [undoStack, setUndoStack] = useState([]);
    const [redoStack, setRedoStack] = useState([]);
    const [textInput, setTextInput] = useState('');
    const [textPos, setTextPos] = useState(null);
    const [bgImage, setBgImage] = useState(null);
    const [saving, setSaving] = useState(false);

    // ─── Load background image ───────────────────────────────────────
    useEffect(() => {
        if (!result) return;
        const img = new Image();
        // Use the first heatmap or original image as background
        const src = result.heatmaps?.wrinkles
            ? `data:image/png;base64,${result.heatmaps.wrinkles}`
            : result.original_image
                ? `data:image/jpeg;base64,${result.original_image}`
                : null;
        if (!src) return;
        img.onload = () => {
            setBgImage(img);
            const canvas = canvasRef.current;
            if (canvas) {
                canvas.width = img.width;
                canvas.height = img.height;
            }
            const overlay = overlayRef.current;
            if (overlay) {
                overlay.width = img.width;
                overlay.height = img.height;
            }
        };
        img.src = src;
    }, [result]);

    // ─── Redraw all layers ───────────────────────────────────────────
    const redrawAll = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Background
        if (bgImage) {
            ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
        }

        // Committed annotations
        annotations.forEach(ann => renderAnnotation(ctx, ann));
    }, [bgImage, annotations]);

    useEffect(() => { redrawAll(); }, [redrawAll]);

    // ─── Get position relative to canvas ─────────────────────────────
    const getPos = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const scaleX = canvasRef.current.width / rect.width;
        const scaleY = canvasRef.current.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    };

    // ─── Render a single annotation to a context ─────────────────────
    const renderAnnotation = (ctx, ann) => {
        ctx.save();
        ctx.strokeStyle = ann.color;
        ctx.fillStyle = ann.color;
        ctx.lineWidth = ann.strokeWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        if (ann.tool === 'marker') {
            ctx.globalAlpha = 0.4;
        }

        switch (ann.tool) {
            case 'pen':
            case 'marker':
            case 'eraser':
                if (ann.points?.length > 1) {
                    ctx.beginPath();
                    if (ann.tool === 'eraser') {
                        ctx.globalCompositeOperation = 'destination-out';
                        ctx.lineWidth = ann.strokeWidth * 3;
                    }
                    ctx.moveTo(ann.points[0].x, ann.points[0].y);
                    for (let i = 1; i < ann.points.length; i++) {
                        ctx.lineTo(ann.points[i].x, ann.points[i].y);
                    }
                    ctx.stroke();
                }
                break;

            case 'line':
                ctx.beginPath();
                ctx.moveTo(ann.start.x, ann.start.y);
                ctx.lineTo(ann.end.x, ann.end.y);
                ctx.stroke();
                break;

            case 'arrow': {
                const dx = ann.end.x - ann.start.x;
                const dy = ann.end.y - ann.start.y;
                const angle = Math.atan2(dy, dx);
                const headLen = Math.max(10, ann.strokeWidth * 4);

                ctx.beginPath();
                ctx.moveTo(ann.start.x, ann.start.y);
                ctx.lineTo(ann.end.x, ann.end.y);
                ctx.stroke();

                // Arrowhead
                ctx.beginPath();
                ctx.moveTo(ann.end.x, ann.end.y);
                ctx.lineTo(ann.end.x - headLen * Math.cos(angle - Math.PI / 6), ann.end.y - headLen * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(ann.end.x - headLen * Math.cos(angle + Math.PI / 6), ann.end.y - headLen * Math.sin(angle + Math.PI / 6));
                ctx.closePath();
                ctx.fill();
                break;
            }

            case 'rect':
                ctx.strokeRect(ann.start.x, ann.start.y, ann.end.x - ann.start.x, ann.end.y - ann.start.y);
                break;

            case 'circle': {
                const rx = Math.abs(ann.end.x - ann.start.x) / 2;
                const ry = Math.abs(ann.end.y - ann.start.y) / 2;
                const cx = ann.start.x + (ann.end.x - ann.start.x) / 2;
                const cy = ann.start.y + (ann.end.y - ann.start.y) / 2;
                ctx.beginPath();
                ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
                ctx.stroke();
                break;
            }

            case 'text':
                ctx.font = `${ann.strokeWidth * 6}px sans-serif`;
                ctx.fillText(ann.text, ann.start.x, ann.start.y);
                break;

            default:
                break;
        }
        ctx.restore();
    };

    // ─── Mouse handlers ──────────────────────────────────────────────
    const handleMouseDown = (e) => {
        const pos = getPos(e);

        if (tool === 'text') {
            setTextPos(pos);
            return;
        }

        setDrawing(true);
        setStartPos(pos);

        if (['pen', 'marker', 'eraser'].includes(tool)) {
            // Start freehand — draw on overlay
            const overlay = overlayRef.current;
            const ctx = overlay.getContext('2d');
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        }
    };

    const handleMouseMove = (e) => {
        if (!drawing) return;
        const pos = getPos(e);
        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');

        if (['pen', 'marker', 'eraser'].includes(tool)) {
            ctx.strokeStyle = tool === 'eraser' ? '#888' : color;
            ctx.lineWidth = tool === 'eraser' ? strokeWidth * 3 : strokeWidth;
            ctx.globalAlpha = tool === 'marker' ? 0.4 : 1;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        } else {
            // Shape preview
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            renderAnnotation(ctx, {
                tool,
                color,
                strokeWidth,
                start: startPos,
                end: pos,
            });
        }
    };

    const handleMouseUp = (e) => {
        if (!drawing) return;
        setDrawing(false);
        const pos = getPos(e);

        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        // Build annotation object
        let ann;
        if (['pen', 'marker', 'eraser'].includes(tool)) {
            // Collect freehand path from overlay (we stored nothing—rebuild from events)
            // Simplified: store start + end; in production, collect points array during move
            ann = {
                tool,
                color,
                strokeWidth,
                points: [startPos, pos],
            };
        } else {
            ann = {
                tool,
                color,
                strokeWidth,
                start: startPos,
                end: pos,
            };
        }

        commitAnnotation(ann);
    };

    // We need to track freehand points properly
    const pointsRef = useRef([]);

    const handleMouseDownImproved = (e) => {
        const pos = getPos(e);

        if (tool === 'text') {
            setTextPos(pos);
            return;
        }

        setDrawing(true);
        setStartPos(pos);
        pointsRef.current = [pos];

        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
    };

    const handleMouseMoveImproved = (e) => {
        if (!drawing) return;
        const pos = getPos(e);
        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');

        if (['pen', 'marker', 'eraser'].includes(tool)) {
            pointsRef.current.push(pos);
            ctx.strokeStyle = tool === 'eraser' ? '#888' : color;
            ctx.lineWidth = tool === 'eraser' ? strokeWidth * 3 : strokeWidth;
            ctx.globalAlpha = tool === 'marker' ? 0.4 : 1;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.beginPath();
            const pts = pointsRef.current;
            if (pts.length >= 2) {
                ctx.moveTo(pts[pts.length - 2].x, pts[pts.length - 2].y);
                ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
            }
            ctx.stroke();
        } else {
            ctx.clearRect(0, 0, overlay.width, overlay.height);
            renderAnnotation(ctx, {
                tool, color, strokeWidth,
                start: startPos,
                end: pos,
            });
        }
    };

    const handleMouseUpImproved = (e) => {
        if (!drawing) return;
        setDrawing(false);
        const pos = getPos(e);

        const overlay = overlayRef.current;
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);

        let ann;
        if (['pen', 'marker', 'eraser'].includes(tool)) {
            pointsRef.current.push(pos);
            ann = {
                tool, color, strokeWidth,
                points: [...pointsRef.current],
            };
        } else {
            ann = {
                tool, color, strokeWidth,
                start: startPos,
                end: pos,
            };
        }
        commitAnnotation(ann);
        pointsRef.current = [];
    };

    // ─── Commit text ─────────────────────────────────────────────────
    const commitText = () => {
        if (!textInput || !textPos) return;
        const ann = {
            tool: 'text',
            color,
            strokeWidth,
            start: textPos,
            text: textInput,
        };
        commitAnnotation(ann);
        setTextInput('');
        setTextPos(null);
    };

    // ─── Commit with undo support ────────────────────────────────────
    const commitAnnotation = (ann) => {
        setAnnotations(prev => {
            const next = [...prev, ann];
            setUndoStack(u => [...u, prev]);
            setRedoStack([]);
            return next;
        });
    };

    const undo = () => {
        if (undoStack.length === 0) return;
        setRedoStack(r => [...r, annotations]);
        setAnnotations(undoStack[undoStack.length - 1]);
        setUndoStack(u => u.slice(0, -1));
    };

    const redo = () => {
        if (redoStack.length === 0) return;
        setUndoStack(u => [...u, annotations]);
        setAnnotations(redoStack[redoStack.length - 1]);
        setRedoStack(r => r.slice(0, -1));
    };

    const clearAll = () => {
        setUndoStack(u => [...u, annotations]);
        setRedoStack([]);
        setAnnotations([]);
    };

    // ─── Export as PNG ───────────────────────────────────────────────
    const exportPNG = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const link = document.createElement('a');
        link.download = `annotated_${sessionId || 'face'}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
    };

    // ─── Save to server ──────────────────────────────────────────────
    const saveAnnotations = async () => {
        if (!sessionId || !serverUrl) return;
        setSaving(true);
        try {
            const payload = annotations.map(ann => ({
                ...ann,
                type: ann.tool,
            }));
            await fetch(`${serverUrl}/sessions/${sessionId}/annotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ annotations: payload }),
            });
        } catch (err) {
            console.error('Save annotations failed:', err);
        } finally {
            setSaving(false);
        }
    };

    // ─── Keyboard shortcuts ──────────────────────────────────────────
    useEffect(() => {
        const handler = (e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === 'z') {
                e.preventDefault();
                if (e.shiftKey) redo();
                else undo();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [annotations, undoStack, redoStack]);

    if (!result) {
        return (
            <div className="flex items-center justify-center h-full text-gray-500">
                <div className="text-center">
                    <p className="text-5xl mb-4">🎨</p>
                    <p>Run an analysis first, then annotate the results here</p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-full" ref={containerRef}>
            {/* ── Toolbar ── */}
            <div className="w-16 bg-gray-900 border-r border-gray-800 flex flex-col items-center py-4 gap-2">
                {TOOLS.map(t => (
                    <button
                        key={t.id}
                        onClick={() => setTool(t.id)}
                        title={t.label}
                        className={`w-10 h-10 rounded-lg flex items-center justify-center text-lg transition
                            ${tool === t.id ? 'bg-indigo-600 ring-2 ring-indigo-400' : 'bg-gray-800 hover:bg-gray-700'}`}
                    >
                        {t.icon}
                    </button>
                ))}

                <hr className="w-8 border-gray-700 my-2" />

                {/* Colors */}
                <div className="flex flex-col gap-1">
                    {COLORS.map(c => (
                        <button
                            key={c}
                            onClick={() => setColor(c)}
                            className={`w-6 h-6 rounded-full border-2 transition mx-auto
                                ${color === c ? 'border-white scale-125' : 'border-transparent'}`}
                            style={{ backgroundColor: c }}
                        />
                    ))}
                </div>

                <hr className="w-8 border-gray-700 my-2" />

                {/* Stroke width */}
                {STROKE_WIDTHS.map(w => (
                    <button
                        key={w}
                        onClick={() => setStrokeWidth(w)}
                        className={`w-10 h-6 flex items-center justify-center rounded transition
                            ${strokeWidth === w ? 'bg-indigo-600' : 'bg-gray-800 hover:bg-gray-700'}`}
                    >
                        <span className="rounded-full bg-white" style={{ width: w * 2, height: w * 2 }} />
                    </button>
                ))}

                <hr className="w-8 border-gray-700 my-2" />

                {/* Actions */}
                <button onClick={undo} title="Undo" className="w-10 h-10 bg-gray-800 hover:bg-gray-700 rounded-lg flex items-center justify-center text-sm" disabled={undoStack.length === 0}>↩</button>
                <button onClick={redo} title="Redo" className="w-10 h-10 bg-gray-800 hover:bg-gray-700 rounded-lg flex items-center justify-center text-sm" disabled={redoStack.length === 0}>↪</button>
                <button onClick={clearAll} title="Clear All" className="w-10 h-10 bg-gray-800 hover:bg-red-900 rounded-lg flex items-center justify-center text-sm">🗑️</button>
            </div>

            {/* ── Canvas Area ── */}
            <div className="flex-1 flex flex-col">
                <div className="flex-1 overflow-auto flex items-center justify-center bg-gray-950 p-4 relative">
                    <div className="relative inline-block" style={{ maxWidth: '100%', maxHeight: '100%' }}>
                        <canvas
                            ref={canvasRef}
                            className="block max-w-full max-h-[70vh] rounded-xl shadow-2xl"
                            style={{ imageRendering: 'auto' }}
                        />
                        <canvas
                            ref={overlayRef}
                            className="absolute top-0 left-0 w-full h-full rounded-xl cursor-crosshair"
                            onMouseDown={handleMouseDownImproved}
                            onMouseMove={handleMouseMoveImproved}
                            onMouseUp={handleMouseUpImproved}
                            onMouseLeave={() => { if (drawing) handleMouseUpImproved({ clientX: 0, clientY: 0 }); }}
                        />
                    </div>

                    {/* Text input floating box */}
                    {textPos && (
                        <div className="absolute bg-gray-800 rounded-lg shadow-lg p-2 flex gap-2" style={{ left: 20, top: 20 }}>
                            <input
                                autoFocus
                                value={textInput}
                                onChange={e => setTextInput(e.target.value)}
                                onKeyDown={e => { if (e.key === 'Enter') commitText(); if (e.key === 'Escape') setTextPos(null); }}
                                placeholder="Type annotation..."
                                className="bg-gray-700 px-3 py-1 rounded text-sm outline-none"
                            />
                            <button onClick={commitText} className="px-3 py-1 bg-indigo-600 rounded text-sm">Add</button>
                            <button onClick={() => setTextPos(null)} className="px-2 py-1 bg-gray-600 rounded text-sm">✕</button>
                        </div>
                    )}
                </div>

                {/* ── Bottom bar ── */}
                <div className="flex items-center gap-3 p-3 bg-gray-900 border-t border-gray-800">
                    <span className="text-xs text-gray-500">
                        {annotations.length} annotation{annotations.length !== 1 ? 's' : ''} • {tool} tool
                    </span>
                    <div className="flex-1" />
                    <button
                        onClick={saveAnnotations}
                        disabled={saving || !sessionId}
                        className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-700 rounded-lg text-sm font-medium disabled:opacity-40"
                    >
                        {saving ? '💾 Saving...' : '💾 Save'}
                    </button>
                    <button
                        onClick={exportPNG}
                        className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 rounded-lg text-sm font-medium"
                    >
                        📥 Export PNG
                    </button>
                </div>
            </div>
        </div>
    );
}
