import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import QUALITY_PRESETS from '../qualityPresets';

/**
 * YouTube-style quality selector dropdown.
 * Uses a React portal so the popup escapes any overflow-hidden ancestors.
 */
function QualitySelector({ quality, onChange, disabled }) {
    const [isOpen, setIsOpen] = useState(false);
    const buttonRef = useRef(null);
    const popupRef = useRef(null);
    const [popupStyle, setPopupStyle] = useState({});

    // Position the portal popup relative to the trigger button
    useEffect(() => {
        if (!isOpen || !buttonRef.current) return;
        const rect = buttonRef.current.getBoundingClientRect();
        setPopupStyle({
            position: 'fixed',
            top: rect.bottom + 6,
            left: Math.max(8, rect.right - 224), // 224 = w-56 = 14rem
            zIndex: 9999,
        });
    }, [isOpen]);

    // Close on outside click or Escape
    useEffect(() => {
        if (!isOpen) return;
        const handleClick = (e) => {
            if (
                buttonRef.current && !buttonRef.current.contains(e.target) &&
                popupRef.current && !popupRef.current.contains(e.target)
            ) {
                setIsOpen(false);
            }
        };
        const handleKey = (e) => { if (e.key === 'Escape') setIsOpen(false); };
        document.addEventListener('mousedown', handleClick);
        document.addEventListener('keydown', handleKey);
        return () => {
            document.removeEventListener('mousedown', handleClick);
            document.removeEventListener('keydown', handleKey);
        };
    }, [isOpen]);

    const currentPreset = QUALITY_PRESETS[quality];
    const presetKeys = Object.keys(QUALITY_PRESETS);

    return (
        <div className="inline-block">
            {/* Trigger button — gear icon + current quality */}
            <button
                ref={buttonRef}
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className="flex items-center gap-1.5 px-2.5 py-1.5 bg-gray-800/80 hover:bg-gray-700 disabled:opacity-50 text-white text-xs rounded-lg transition-colors border border-gray-600"
                title="Quality settings"
            >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                <span className="font-medium">{currentPreset?.label || quality}</span>
                {currentPreset?.icon === 'HD' && (
                    <span className="text-[9px] font-bold bg-blue-500 text-white px-1 rounded">HD</span>
                )}
            </button>

            {/* Dropdown popup — rendered via portal to escape overflow-hidden ancestors */}
            {isOpen && createPortal(
                <div
                    ref={popupRef}
                    style={popupStyle}
                    className="w-56 bg-gray-900 border border-gray-600 rounded-lg shadow-2xl overflow-hidden"
                >
                    {/* Header */}
                    <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between">
                        <span className="text-xs font-semibold text-gray-300">Quality</span>
                        <button
                            onClick={() => setIsOpen(false)}
                            className="text-gray-400 hover:text-white"
                        >
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                            </svg>
                        </button>
                    </div>

                    {/* Options */}
                    <div className="py-1">
                        {presetKeys.map((key) => {
                            const preset = QUALITY_PRESETS[key];
                            const isSelected = key === quality;
                            return (
                                <button
                                    key={key}
                                    onClick={() => { onChange(key); setIsOpen(false); }}
                                    className={`w-full px-3 py-2 flex items-center justify-between text-left hover:bg-gray-700/60 transition-colors ${isSelected ? 'bg-gray-700/40' : ''}`}
                                >
                                    <div className="flex items-center gap-2">
                                        <span className={`w-1.5 h-1.5 rounded-full ${isSelected ? 'bg-blue-400' : 'bg-transparent'}`} />
                                        <span className={`text-sm ${isSelected ? 'text-white font-medium' : 'text-gray-300'}`}>
                                            {preset.label}
                                        </span>
                                        {preset.icon === 'HD' && (
                                            <span className="text-[8px] font-bold bg-blue-500/80 text-white px-1 rounded">HD</span>
                                        )}
                                    </div>
                                    <span className="text-[10px] text-gray-500">
                                        {key === 'auto' ? '' : `${preset.fps}fps · ${preset.width}×${preset.height}`}
                                    </span>
                                </button>
                            );
                        })}
                    </div>
                </div>,
                document.body
            )}
        </div>
    );
}

export default QualitySelector;
