import React, { useState } from 'react';

const SESSION_API = 'https://slot-booking-app.azurewebsites.net/api/session/check';

function BookingCheck({ onSuccess }) {
    const [code, setCode] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        const trimmed = code.trim();
        if (!trimmed) {
            setError('Please enter your booking code.');
            return;
        }

        setError('');
        setLoading(true);

        try {
            const res = await fetch(SESSION_API, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: trimmed }),
            });

            const data = await res.json();

            if (res.ok && data.status === 'ok') {
                onSuccess();
                return;
            }

            // Map API error messages to something clean
            if (data.error) {
                if (data.startsAt) {
                    const when = new Date(data.startsAt).toLocaleString();
                    setError(`${data.error} Your session starts at ${when}.`);
                } else {
                    setError(data.error);
                }
            } else {
                setError('Something went wrong. Please try again.');
            }
        } catch {
            setError('Unable to reach the server. Check your internet connection.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
            <div className="w-full max-w-md p-8 bg-gray-800 rounded-lg shadow-lg">
                <h2 className="text-3xl font-bold text-center mb-2">Doctor Preview</h2>
                <p className="text-center text-gray-400 mb-8 text-sm">Enter your booking code to continue</p>

                {error && (
                    <div className="bg-red-500 bg-opacity-20 border border-red-500 text-red-200 px-4 py-3 rounded mb-6 text-sm text-center">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                        <label htmlFor="booking-code" className="block text-sm font-medium text-gray-400 mb-2">
                            Booking Code
                        </label>
                        <input
                            id="booking-code"
                            type="text"
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white text-center text-lg tracking-widest placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
                            placeholder="SB-XXXXXXXX"
                            value={code}
                            onChange={(e) => setCode(e.target.value.toUpperCase())}
                            disabled={loading}
                            autoFocus
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-lg transition duration-200"
                    >
                        {loading ? 'Validating...' : 'Enter'}
                    </button>

                    <p className="text-xs text-center text-gray-500 mt-4">
                        Authorized personnel only. Contact support if you don't have a code.
                    </p>
                </form>
            </div>
        </div>
    );
}

export default BookingCheck;
