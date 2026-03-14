import React from 'react';

/**
 * HeatmapOverlay — Renders a heatmap as a semi-transparent canvas layer
 * on top of the original face image.
 */

export default function HeatmapOverlay({ imageBase64, label }) {
    if (!imageBase64) return null;

    return (
        <div className="space-y-2">
            <div className="relative rounded-xl overflow-hidden border border-gray-700">
                <img
                    src={`data:image/png;base64,${imageBase64}`}
                    alt={`${label} heatmap`}
                    className="w-full max-h-[500px] object-contain bg-black"
                />
                <div className="absolute top-2 left-2 bg-black/70 px-2.5 py-1 rounded-lg text-[10px] font-semibold capitalize text-gray-200">
                    {label} Map
                </div>
            </div>
            <div className="flex items-center justify-center gap-5 text-[10px] text-gray-500">
                {[['#3b82f6', 'Healthy'], ['#22c55e', 'Mild'], ['#eab308', 'Moderate'], ['#ef4444', 'Concern']].map(([c, l]) => (
                    <div key={l} className="flex items-center gap-1">
                        <div className="w-3 h-2 rounded-sm" style={{ background: c }} />
                        <span>{l}</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
