import React, { useState, useEffect } from 'react';
import { useSignIn } from '@clerk/clerk-react';

function Login({ onLogin }) {
    const HARDCODED_EMAIL = 'code.abc97@gmail.com';
    const { isLoaded, signIn, setActive } = useSignIn();
    const [otp, setOtp] = useState('');
    const [error, setError] = useState('');
    const [otpSent, setOtpSent] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [isVerifying, setIsVerifying] = useState(false);

    useEffect(() => {
        console.log('🔐 Login component mounted with Clerk integration');
        console.log('Configured email:', HARDCODED_EMAIL);
        console.log('Clerk loaded:', isLoaded);
    }, [isLoaded]);

    useEffect(() => {
        if (otp) {
            console.log('OTP input changed:', otp, '(length:', otp.length, ')');
        }
    }, [otp]);

    const handleSendOtp = async () => {
        console.log('=== SEND OTP CLICKED ===');
        console.log('Sending OTP to:', HARDCODED_EMAIL);
        console.log('Timestamp:', new Date().toISOString());
        
        if (!isLoaded) {
            console.error('❌ Clerk not loaded yet');
            setError('Authentication service is loading...');
            return;
        }
        
        setError('');
        setSuccessMessage('');
        
        try {
            console.log('Creating Clerk sign-in with email code strategy...');
            
            // Start the sign-in process with email code strategy
            await signIn.create({
                identifier: HARDCODED_EMAIL,
            });

            // Send the email code
            await signIn.prepareFirstFactor({
                strategy: 'email_code',
                emailAddressId: signIn.supportedFirstFactors.find(
                    (factor) => factor.strategy === 'email_code'
                )?.emailAddressId,
            });

            console.log('✅ OTP sent successfully via Clerk');
            setOtpSent(true);
            setSuccessMessage('OTP has been sent.');
            
        } catch (err) {
            console.error('❌ SEND OTP ERROR:', err);
            console.error('Error details:', err.errors);
            
            const errorMessage = err.errors?.[0]?.message || err.message || 'Failed to send OTP';
            setError(errorMessage);
        }
        
        console.log('=== SEND OTP COMPLETE ===');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        console.log('=== LOGIN SUBMIT ===');
        console.log('OTP entered:', otp);
        console.log('OTP sent status:', otpSent);
        
        if (!isLoaded) {
            console.error('❌ Clerk not loaded yet');
            setError('Authentication service is loading...');
            return;
        }
        
        setError('');

        if (!otpSent) {
            console.log('❌ Login failed: OTP not requested yet');
            setError('Please request an OTP first');
            return;
        }

        if (otp.length !== 6 || isNaN(otp)) {
            console.log('❌ Login failed: Invalid OTP format');
            console.log('OTP length:', otp.length, 'Expected: 6');
            console.log('Is numeric:', !isNaN(otp));
            setError('Please enter a valid 6-digit OTP');
            return;
        }

        console.log('✅ OTP format valid, verifying with Clerk...');
        setIsVerifying(true);
        
        try {
            // Verify the OTP code
            const result = await signIn.attemptFirstFactor({
                strategy: 'email_code',
                code: otp,
            });

            if (result.status === 'complete') {
                console.log('✅ OTP verified successfully');
                console.log('Setting active session...');
                
                // Set the active session
                await setActive({ session: result.createdSessionId });
                
                console.log('✅ Session activated');
                console.log('Logging in user...');
                onLogin();
                console.log('=== LOGIN COMPLETE ===');
            } else {
                console.log('⚠️ Sign-in not complete, status:', result.status);
                setError('Authentication incomplete. Please try again.');
            }
        } catch (err) {
            console.error('❌ OTP VERIFICATION ERROR:', err);
            console.error('Error details:', err.errors);
            
            const errorMessage = err.errors?.[0]?.message || err.message || 'Invalid OTP code';
            setError(errorMessage);
        } finally {
            setIsVerifying(false);
        }
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

                {successMessage && (
                    <div className="bg-green-500 bg-opacity-20 border border-green-500 text-green-100 px-4 py-3 rounded mb-4 text-sm text-center">
                        {successMessage}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-6">
                    <div>
                        <button
                            type="button"
                            onClick={handleSendOtp}
                            disabled={!isLoaded || otpSent}
                            className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold rounded-lg transition duration-200"
                        >
                            {otpSent ? 'OTP Sent ✓' : 'Send OTP'}
                        </button>
                    </div>

                    <div>
                        <label htmlFor="otp" className="block text-sm font-medium text-gray-400 mb-2">
                            One-Time Password (OTP)
                        </label>
                        <input
                            type="text"
                            id="otp"
                            maxLength="6"
                            className={`w-full px-4 py-3 border rounded-lg text-center text-lg tracking-widest ${
                                otpSent 
                                    ? 'bg-gray-700 border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-white placeholder-gray-500' 
                                    : 'bg-gray-600 border-gray-600 text-gray-400 cursor-not-allowed'
                            }`}
                            placeholder="••••••"
                            value={otp}
                            onChange={(e) => setOtp(e.target.value)}
                            disabled={!otpSent}
                            required
                        />
                    </div>

                    <button
                        type="submit"
                        disabled={isVerifying || !isLoaded}
                        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-lg transition duration-200"
                    >
                        {isVerifying ? 'Verifying...' : 'Login'}
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
