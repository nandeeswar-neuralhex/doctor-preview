import React from 'react';

/**
 * FaceMeasurements — Compact measurement table with facial thirds bar.
 */

export default function FaceMeasurements({ measurements }) {
    if (!measurements || Object.keys(measurements).length === 0) return null;

    const thirds = measurements.facial_thirds;

    return (
        <div className="mt-2 space-y-2">
            <table className="w-full text-[10px]">
                <tbody>
                    {Object.entries(measurements).map(([key, data]) => {
                        if (!data || typeof data !== 'object' || !('value' in data)) return null;
                        return (
                            <tr key={key} className="border-b border-gray-800/30">
                                <td className="py-1 text-gray-400 capitalize">{key.replace(/_/g, ' ')}</td>
                                <td className="py-1 text-right font-mono text-white">{data.value}{data.unit === 'degrees' ? '°' : ''}</td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>

            {thirds && (
                <div>
                    <p className="text-[8px] text-gray-600 mb-1">Facial Thirds</p>
                    <div className="flex gap-0.5 h-4 rounded overflow-hidden">
                        <div className="bg-indigo-600 flex items-center justify-center text-[7px] text-white" style={{ width: `${thirds.upper}%` }}>{thirds.upper?.toFixed(0)}%</div>
                        <div className="bg-purple-600 flex items-center justify-center text-[7px] text-white" style={{ width: `${thirds.middle}%` }}>{thirds.middle?.toFixed(0)}%</div>
                        <div className="bg-pink-600 flex items-center justify-center text-[7px] text-white" style={{ width: `${thirds.lower}%` }}>{thirds.lower?.toFixed(0)}%</div>
                    </div>
                    <div className="flex justify-between text-[7px] text-gray-600 mt-0.5">
                        <span>Upper</span><span>Middle</span><span>Lower</span>
                    </div>
                </div>
            )}
        </div>
    );
}
