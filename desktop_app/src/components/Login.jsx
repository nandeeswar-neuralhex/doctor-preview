import React, { useState } from 'react';

function Login({ onLogin }) {
    const [email, setEmail] = useState('');
    const [otp, setOtp] = useState('');
    const [error, setError] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        setError('');

        if (email !== 'deeve@gmail.com') {
            setError('Invalid email address');
            return;
        }

        if (otp.length !== 6 || isNaN(otp)) {
            setError('Please enter a valid 6-digit OTP');
            return;
        }

        onLogin();
    };

    return (
        <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
            <div className="w-full max-w-md p-8 bg-gray-800 rounded-lg shadow-lg">
                <h2 className="text-3xl font-bold text-center mb-6">Doctor Preview Login</h2>

                {error && (
                    <div className="bg-red-500 bg-opacity-20 border border-red-500 text-red-100 px-4 py-3 rounded mb-4 text-sm text-center">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <label htmlFor="email" className="block text-sm font-medium text-gray-400 mb-2">
                            Email Address
                        </label>
                        <input
                            type="email"
                            id="email"
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-gray-500"
                            placeholder="Enter your email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    <div>
                        <label htmlFor="otp" className="block text-sm font-medium text-gray-400 mb-2">
                            One-Time Password (OTP)
                        </label>
                        <input
                            type="text"
                            id="otp"
                            maxLength="6"
                            className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-gray-500 tracking-widest text-center text-lg"
                            placeholder="••••••"
                            value={otp}
                            onChange={(e) => setOtp(e.target.value)}
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200"
                    >
                        Login
                    </button>

                    <p className="text-xs text-center text-gray-500 mt-4">
                        Authorized personnel only.
                    </p>
                </form>
            </div>
        </div>
    );
}

export default Login;
