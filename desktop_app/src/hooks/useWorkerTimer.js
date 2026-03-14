import { useRef, useCallback, useEffect } from 'react';

/**
 * Web Worker–based interval timer that is NOT throttled when the browser tab
 * is in the background.
 *
 * Chrome has 3 levels of background restrictions:
 *  1. Timer throttling (hidden tab): setTimeout → 1s min.  ← Worker bypasses this
 *  2. Page freeze (hidden >5min):    ALL JS paused.         ← Web Lock prevents this
 *  3. Tab discard (memory pressure):  Tab unloaded.          ← We can't prevent this
 *
 * This hook handles levels 1 + 2 via:
 *  - A dedicated Worker thread for timing (immune to timer throttling)
 *  - navigator.locks.request() to hold a Web Lock (prevents page freeze)
 */
export default function useWorkerTimer() {
    const workerRef = useRef(null);
    const cbRef = useRef(null);
    const lockRef = useRef(null);

    /** Spin up a fresh Worker (inline via Blob URL) */
    const createWorker = () => {
        const code = `
            let tid = null;
            self.onmessage = (e) => {
                if (e.data.cmd === 'start') {
                    if (tid !== null) clearInterval(tid);
                    tid = setInterval(() => self.postMessage('tick'), e.data.interval);
                } else if (e.data.cmd === 'interval') {
                    if (tid !== null) clearInterval(tid);
                    tid = setInterval(() => self.postMessage('tick'), e.data.interval);
                } else if (e.data.cmd === 'stop') {
                    if (tid !== null) { clearInterval(tid); tid = null; }
                }
            };
        `;
        const blob = new Blob([code], { type: 'application/javascript' });
        const url = URL.createObjectURL(blob);
        const w = new Worker(url);
        URL.revokeObjectURL(url);
        w.onmessage = () => {
            if (cbRef.current) cbRef.current();
        };
        return w;
    };

    /**
     * Acquire a Web Lock to prevent Chrome from freezing the page.
     * The lock is held as long as the returned AbortController is not aborted.
     * navigator.locks.request() with an infinite-wait callback keeps the lock alive.
     */
    const acquireWebLock = () => {
        if (!navigator.locks) return;  // Not supported (Safari <15.4, Firefox <96)
        const ac = new AbortController();
        navigator.locks.request(
            'doctor-preview-background-keepalive',
            { signal: ac.signal },
            () => new Promise((resolve) => {
                // Hold the lock forever (until we abort).
                // Store resolve so we can release on stop().
                lockRef.current = { resolve, ac };
            })
        ).catch(() => {});  // AbortError when we release — expected
        return ac;
    };

    const start = useCallback((intervalMs, callback) => {
        stop();  // kill any previous worker + lock
        cbRef.current = callback;
        const w = createWorker();
        workerRef.current = w;
        w.postMessage({ cmd: 'start', interval: intervalMs });
        acquireWebLock();
        console.log('[WorkerTimer] Started:', intervalMs, 'ms interval + Web Lock acquired');
    }, []);

    const stop = useCallback(() => {
        if (workerRef.current) {
            workerRef.current.postMessage({ cmd: 'stop' });
            workerRef.current.terminate();
            workerRef.current = null;
        }
        cbRef.current = null;
        // Release Web Lock
        if (lockRef.current) {
            lockRef.current.resolve();       // release the lock promise
            lockRef.current.ac.abort();      // abort the lock request
            lockRef.current = null;
        }
        console.log('[WorkerTimer] Stopped');
    }, []);

    /** Change the tick interval without restarting */
    const setIntervalMs = useCallback((ms) => {
        if (workerRef.current) {
            workerRef.current.postMessage({ cmd: 'interval', interval: ms });
        }
    }, []);

    // Clean up on unmount
    useEffect(() => {
        return () => stop();
    }, [stop]);

    return { start, stop, setInterval: setIntervalMs };
}
