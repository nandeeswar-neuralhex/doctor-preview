import React, { useState } from 'react';
import ImageUpload from './components/ImageUpload';
import CameraView from './components/CameraView';
import Settings from './components/Settings';
import Login from './components/Login';

function App() {
    const [serverUrl, setServerUrl] = useState('https://u4ln48a51ahjsh-8765.proxy.runpod.net');
    const [targetImages, setTargetImages] = useState([]);
    const [selectedImageIndex, setSelectedImageIndex] = useState(0);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isLoggedIn, setIsLoggedIn] = useState(false);

    const handleLogin = () => {
        setIsLoggedIn(true);
    };

    const handleLogout = () => {
        setIsLoggedIn(false);
        // Optional: clear any other state if needed
    };

    if (!isLoggedIn) {
        return <Login onLogin={handleLogin} />;
    }

    return (
        <div className="h-screen flex flex-col bg-gray-900">
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-white">Doctor Preview</h1>
                        <p className="text-sm text-gray-400">Real-time Surgery Preview System</p>
                    </div>
                    <div className="flex items-center gap-4">
                        <Settings serverUrl={serverUrl} setServerUrl={setServerUrl} />
                        <button
                            onClick={handleLogout}
                            className="text-sm text-gray-400 hover:text-white transition-colors border border-gray-600 px-3 py-1 rounded hover:bg-gray-700"
                        >
                            Logout
                        </button>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Left Sidebar - Image Upload */}
                <aside className="w-80 bg-gray-800 border-r border-gray-700 p-6 overflow-y-auto">
                    <ImageUpload
                        targetImages={targetImages}
                        setTargetImages={setTargetImages}
                        selectedImageIndex={selectedImageIndex}
                        setSelectedImageIndex={setSelectedImageIndex}
                        serverUrl={serverUrl}
                    />
                </aside>

                {/* Main Area - Camera View */}
                <main className="flex-1 p-6">
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
            <footer className="bg-gray-800 border-t border-gray-700 px-6 py-3">
                <div className="flex items-center justify-between text-sm text-gray-400">
                    <div>
                        {serverUrl ? (
                            <span className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                                Connected to {serverUrl}
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
    );
}

export default App;
