import { useState, useCallback } from 'react';

function useWebcam(withAudio = false) {
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
                audio: withAudio
            });
            setStream(mediaStream);
            setError(null);
            return mediaStream;
        } catch (err) {
            setError(`Failed to access webcam: ${err.message}`);
            console.error('Webcam error:', err);
            return null;
        }
    }, [withAudio]);

    const stopWebcam = useCallback(() => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            setStream(null);
        }
    }, [stream]);

    return { stream, error, startWebcam, stopWebcam };
}

export default useWebcam;
