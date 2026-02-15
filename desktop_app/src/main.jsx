import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

// Error boundary to catch React crashes and show them instead of blank screen
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    componentDidCatch(error, info) {
        console.error('[React Crash]', error, info.componentStack);
    }
    render() {
        if (this.state.hasError) {
            return React.createElement('div', {
                style: { padding: '40px', color: '#ff6b6b', background: '#1a1a1a', minHeight: '100vh', fontFamily: 'monospace' }
            },
                React.createElement('h1', null, '⚠️ App Crashed'),
                React.createElement('pre', { style: { whiteSpace: 'pre-wrap', marginTop: '20px', color: '#ffa' } },
                    String(this.state.error)
                ),
                React.createElement('button', {
                    onClick: () => window.location.reload(),
                    style: { marginTop: '20px', padding: '10px 20px', background: '#4a9eff', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }
                }, 'Reload App')
            );
        }
        return this.props.children;
    }
}

ReactDOM.createRoot(document.getElementById('root')).render(
    <ErrorBoundary>
        <App />
    </ErrorBoundary>
);
