import React, { useState, useEffect } from 'react';
import { useAuth, useClerk } from '@clerk/clerk-react';
import ImageUpload from './components/ImageUpload';
import CameraView from './components/CameraView';
import FaceAnalysis from './components/FaceAnalysis';
import Login from './components/Login';

const SESSION_DURATION_MS = 90 * 60 * 1000; // 90 minutes
const SESSION_START_KEY = 'dp_session_start';

function App() {
    const { isSignedIn, isLoaded } = useAuth();
    const { signOut } = useClerk();
    // Track whether the startup sign-out has completed.
    // Every app launch must start with a fresh login — if Clerk finds a stale
    // session from a previous run (e.g. Cmd+Q killed the app before the
    // renderer could sign out), we sign out immediately and show the login screen.
    const [startupLogoutDone, setStartupLogoutDone] = useState(false);
    // Server URL is now hidden from user - loaded from environment variable
    const serverUrl = import.meta.env.VITE_SERVER_URL || 'http://20.9.36.27:8765';
    
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
    const [sessionMinsLeft, setSessionMinsLeft] = useState(null);
    const [activeTab, setActiveTab] = useState('preview'); // 'preview' | 'analysis'

    const handleLogin = () => {
        // This is handled by Clerk now, but kept for compatibility
        console.log('Login successful');
    };

    const handleLogout = async () => {
        console.log('Logging out...');
        sessionStorage.removeItem(SESSION_START_KEY);
        await signOut();
    };

    // ── Startup: force sign-out if a stale Clerk session exists ────────────────
    // This runs once after Clerk loads. If a previous session is still alive
    // (quit, crash, force-kill), we sign out so the user always sees the login.
    useEffect(() => {
        if (!isLoaded || startupLogoutDone) return;
        if (isSignedIn) {
            console.log('[App] Stale Clerk session found on startup — signing out');
            sessionStorage.removeItem(SESSION_START_KEY);
            signOut().finally(() => setStartupLogoutDone(true));
        } else {
            setStartupLogoutDone(true);
        }
    }, [isLoaded]); // eslint-disable-line react-hooks/exhaustive-deps
    // ── end startup sign-out ───────────────────────────────────────────────────

    // ── 90-minute session enforcement ──────────────────────────────────────────
    // The login timestamp is written to sessionStorage (cleared on app exit,
    // survives React re-renders). Even if DevTools reloads the page, the
    // timestamp is preserved so the clock cannot be reset without quitting.
    useEffect(() => {
        if (!isSignedIn) return;

        // Record the session start time once per Clerk sign-in.
        // If already set (e.g. page re-render), keep the original timestamp.
        if (!sessionStorage.getItem(SESSION_START_KEY)) {
            sessionStorage.setItem(SESSION_START_KEY, String(Date.now()));
        }

        const checkExpiry = () => {
            const start = parseInt(sessionStorage.getItem(SESSION_START_KEY) || '0', 10);
            const elapsed = Date.now() - start;
            if (elapsed >= SESSION_DURATION_MS) {
                console.warn('[Session] 90-minute limit reached — forcing logout');
                handleLogout();
            }
        };

        // Calculate exact remaining time and set a precise one-shot timeout
        const start = parseInt(sessionStorage.getItem(SESSION_START_KEY) || '0', 10);
        const remaining = SESSION_DURATION_MS - (Date.now() - start);

        if (remaining <= 0) {
            // Already expired (e.g. app was suspended and resumed)
            handleLogout();
            return;
        }

        const timeoutId = setTimeout(() => {
            console.warn('[Session] Auto-logout after 90 minutes');
            handleLogout();
        }, remaining);

        // Belt-and-suspenders: check every 30 s in case the machine was
        // asleep and the timeout fired late (or was throttled).
        // Also update the visible countdown every 30 s.
        const updateCountdown = () => {
            const s = parseInt(sessionStorage.getItem(SESSION_START_KEY) || '0', 10);
            const mins = Math.ceil((SESSION_DURATION_MS - (Date.now() - s)) / 60_000);
            setSessionMinsLeft(mins > 0 ? mins : 0);
        };
        updateCountdown(); // set immediately
        const intervalId = setInterval(() => { checkExpiry(); updateCountdown(); }, 30_000);

        // Also check when the window comes back into focus after being hidden
        const onVisibility = () => {
            if (document.visibilityState === 'visible') checkExpiry();
        };
        document.addEventListener('visibilitychange', onVisibility);

        return () => {
            clearTimeout(timeoutId);
            clearInterval(intervalId);
            document.removeEventListener('visibilitychange', onVisibility);
        };
    }, [isSignedIn]); // eslint-disable-line react-hooks/exhaustive-deps
    // ── end session enforcement ────────────────────────────────────────────────

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

    // Window close / app quit — main sends 'force-logout' before destroying the window.
    // We sign out via Clerk (so the session token is cleared from localStorage),
    // then notify main it's safe to proceed with the close/quit.
    // Works on: red-X close, Cmd+Q, and any programmatic window.close().
    useEffect(() => {
        const handler = async () => {
            console.log('[App] force-logout received — signing out before window closes');
            sessionStorage.removeItem(SESSION_START_KEY);
            try {
                if (isSignedIn) await signOut();
            } catch (e) {
                console.error('[App] signOut error on close:', e);
            } finally {
                window.electronAPI?.signOutComplete?.();
            }
        };
        if (window.electronAPI?.onForceLogout) {
            window.electronAPI.onForceLogout(handler);
        }
        return () => {
            if (window.electronAPI?.removeForceLogout) {
                window.electronAPI.removeForceLogout(handler);
            }
        };
    }, [isSignedIn]); // re-bind when sign-in state changes so handler has fresh value

    // Show loading state while Clerk is initializing or startup sign-out is in progress
    if (!isLoaded || !startupLogoutDone) {
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
            {/* Session expiry warning banner */}
            {sessionMinsLeft !== null && sessionMinsLeft <= 10 && (
                <div className="bg-yellow-600 text-white text-center text-sm py-1 px-4">
                    ⚠️ Session expires in {sessionMinsLeft} minute{sessionMinsLeft !== 1 ? 's' : ''} — save your work.
                </div>
            )}
            {/* Header */}
            <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
                <div className="flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold text-white">Doctor Preview</h1>
                        <p className="text-sm text-gray-400">Real-time Surgery Preview System</p>
                    </div>
                    {/* Module Tabs */}
                    <div className="flex gap-1 bg-gray-900 rounded-lg p-1">
                        <button
                            onClick={() => setActiveTab('preview')}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                activeTab === 'preview'
                                    ? 'bg-indigo-600 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                            }`}
                        >
                            🎥 Live Preview
                        </button>
                        <button
                            onClick={() => setActiveTab('analysis')}
                            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                                activeTab === 'analysis'
                                    ? 'bg-indigo-600 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                            }`}
                        >
                            🔬 Face Analysis
                        </button>
                    </div>
                    <div className="flex items-center gap-4">
                        {sessionMinsLeft !== null && (
                            <span className="text-xs text-gray-500">
                                Session: {sessionMinsLeft}m left
                            </span>
                        )}
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
                {activeTab === 'preview' ? (
                    <>
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
                    </>
                ) : (
                    <main className="flex-1 overflow-y-auto">
                        <FaceAnalysis serverUrl={serverUrl} />
                    </main>
                )}
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
