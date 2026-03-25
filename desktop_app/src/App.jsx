import React, { useState, useRef, useEffect } from 'react';
import ImageUpload from './components/ImageUpload';
import CameraView from './components/CameraView';
import BookingCheck from './components/BookingCheck';

const GPU_OPTIONS = [
    { id: 'prod-t4',   label: 'T4',         url: import.meta.env.VITE_SERVER_URL     || 'https://faceiq.sparkiq.ai' },
    { id: 'legacy-t4', label: 'T4 Legacy',  url: import.meta.env.VITE_GPU_LEGACY_URL || 'https://t4lagcyfaceiq.sparkiq.ai' },
    { id: 'debug-t4',  label: 'T4 Debug',   url: import.meta.env.VITE_GPU_DEBUG_URL  || 'https://faceiq-debug.sparkiq.ai' },
];

/* ── GPU Badge ───────────────────────────────────────────────────────
   Looks like plain text "GPU — T4".  Click to reveal other options.
   Normal users won't notice it's interactive.                       */
function GpuBadge({ options, selected, onSelect, disabled }) {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);

    // close on outside click
    useEffect(() => {
        if (!open) return;
        const close = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
        document.addEventListener('mousedown', close);
        return () => document.removeEventListener('mousedown', close);
    }, [open]);

    return (
        <div ref={ref} className="relative select-none">
            <span
                onClick={() => !disabled && setOpen(o => !o)}
                className={`text-xs font-medium tracking-wide ${disabled ? 'text-gray-600 cursor-not-allowed' : 'text-gray-400 cursor-pointer hover:text-gray-200'} transition-colors`}
            >
                GPU&nbsp;&mdash;&nbsp;{selected.label}
            </span>

            {open && (
                <div className="absolute right-0 top-full mt-1 bg-gray-800 border border-gray-600 rounded-lg shadow-lg z-50 min-w-[140px] py-1">
                    {options.map(gpu => (
                        <button
                            key={gpu.id}
                            onClick={() => { onSelect(gpu); setOpen(false); }}
                            className={`w-full text-left px-3 py-1.5 text-xs transition-colors ${
                                gpu.id === selected.id
                                    ? 'text-white bg-gray-700'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                            }`}
                        >
                            {gpu.label}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

function App() {
    const [sessionValid, setSessionValid] = useState(false);

    const defaultGpu = GPU_OPTIONS[0];
    const [selectedGpu, setSelectedGpu] = useState(defaultGpu);
    const serverUrl = selectedGpu.url;

    // Function to mask URL - shows only last 2 digits of IP
    const getMaskedUrl = (url) => {
        if (!url) return '...';
        const ipMatch = url.match(/(\d+\.\d+\.\d+\.(\d{2,3}))/);
        if (ipMatch) {
            const lastDigits = ipMatch[2].slice(-2);
            return `...${lastDigits}`;
        }
        return '...';
    };
    const [targetImages, setTargetImages] = useState([]);
    const [selectedImageIndex, setSelectedImageIndex] = useState(0);
    const [isStreaming, setIsStreaming] = useState(false);

    return (
        <>
            {!sessionValid && <BookingCheck onSuccess={() => setSessionValid(true)} />}
            {sessionValid && (
        <div className="h-screen flex flex-col bg-gray-900">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-4 py-3 md:px-6 md:py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-xl md:text-2xl font-bold text-white">Doctor Preview</h1>
                        <p className="text-xs md:text-sm text-gray-400">Real-time Surgery Preview System</p>
                    </div>
                    <GpuBadge
                        options={GPU_OPTIONS}
                        selected={selectedGpu}
                        onSelect={setSelectedGpu}
                        disabled={isStreaming}
                    />
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
                {/* Left Sidebar - Image Upload */}
                <aside className="w-full md:w-80 bg-gray-800 border-b md:border-b-0 md:border-r border-gray-700 p-4 md:p-6 overflow-y-auto max-h-48 md:max-h-none">
                    <ImageUpload
                        targetImages={targetImages}
                        setTargetImages={setTargetImages}
                        selectedImageIndex={selectedImageIndex}
                        setSelectedImageIndex={setSelectedImageIndex}
                        serverUrl={serverUrl}
                    />
                </aside>

                {/* Main Area - Camera View */}
                <main className="flex-1 p-3 md:p-6 overflow-y-auto">
                    <CameraView
                        serverUrl={serverUrl}
                        targetImage={targetImages[selectedImageIndex]}
                        allTargetImages={targetImages}
                        isStreaming={isStreaming}
                        setIsStreaming={setIsStreaming}
                    />
                </main>
            </div>

            {/* Footer */}
            <footer className="bg-gray-800 border-t border-gray-700 px-4 py-2 md:px-6 md:py-3">
                <div className="flex items-center justify-between text-sm text-gray-400">
                    <div>
                        {serverUrl ? (
                            <span className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                                Connected to Server {getMaskedUrl(serverUrl)}
                            </span>
                        ) : (
                            <span className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                                Not connected
                            </span>
                        )}
                    </div>
                    <div>
                        {targetImages.length} image{targetImages.length !== 1 ? 's' : ''} uploaded
                    </div>
                </div>
            </footer>
        </div>
            )}
        </>
    );
}

export default App;
