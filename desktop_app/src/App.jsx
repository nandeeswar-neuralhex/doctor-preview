import React, { useState, useEffect } from 'react';
import { useAuth, useClerk } from '@clerk/clerk-react';
import ImageUpload from './components/ImageUpload';
import CameraView from './components/CameraView';
import Login from './components/Login';

function App() {
    const { isSignedIn, isLoaded } = useAuth();
    const { signOut } = useClerk();
    // Server URL is now hidden from user - loaded from environment variable
    const serverUrl = import.meta.env.VITE_SERVER_URL || 'http://20.115.36.199:8765';
    
    // Function to mask URL - shows only last 2 digits of IP
    const getMaskedUrl = (url) => {
        if (!url) return '...';
        // Extract last 2 digits from IP (e.g., 20.115.36.199 -> ...99)
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

    const handleLogin = () => {
        // This is handled by Clerk now, but kept for compatibility
        console.log('Login successful');
    };

    const handleLogout = async () => {
        console.log('Logging out...');
        await signOut();
    };

    // Cmd+R / Cmd+Shift+R triggers logout instead of refresh
    useEffect(() => {
        const handler = () => {
            console.log('Refresh intercepted — logging out...');
            handleLogout();
        };
        if (window.electronAPI?.onTriggerLogout) {
            window.electronAPI.onTriggerLogout(handler);
        }
        return () => {
            if (window.electronAPI?.removeTriggerLogout) {
                window.electronAPI.removeTriggerLogout(handler);
            }
        };
    }, []);

    // Show loading state while Clerk is initializing
    if (!isLoaded) {
        return (
            <div className="h-screen flex items-center justify-center bg-gray-900 text-white">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4"></div>
                    <p>Loading...</p>
                </div>
            </div>
        );
    }

    if (!isSignedIn) {
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
    );
}

export default App;
