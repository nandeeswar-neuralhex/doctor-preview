import { useState, useCallback } from 'react';

function useWebcam() {
    const [stream, setStream] = useState(null);
    const [error, setError] = useState(null);

    const startWebcam = useCallback(async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });
            setStream(mediaStream);
            setError(null);
        } catch (err) {
            setError(`Failed to access webcam: ${err.message}`);
            console.error('Webcam error:', err);
        }
    }, []);

    const stopWebcam = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    }, [stream]);

    return { stream, error, startWebcam, stopWebcam };
}

export default useWebcam;
