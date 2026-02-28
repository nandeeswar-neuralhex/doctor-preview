/**
 * Off-main-thread JPEG encoder using OffscreenCanvas.
 *
 * Why: canvas.toBlob() on the main thread blocks UI, especially on Windows
 * where there's no hardware JPEG encoder. By using OffscreenCanvas.convertToBlob()
 * inside a Worker, encoding runs on a separate thread — zero main-thread jank.
 *
 * Fallback: If OffscreenCanvas isn't available, falls back to main-thread canvas.toBlob().
 *
 * Usage:
 *   const encoder = new FrameEncoder();
 *   const blob = await encoder.encode(videoElement, width, height, quality);
 *   encoder.dispose();
 */

// Web Worker code as a string (inlined to avoid separate file / bundler issues)
const WORKER_CODE = `
    let canvas = null;
    let ctx = null;

    self.onmessage = async (e) => {
        const { id, bitmap, width, height, quality } = e.data;
        try {
            // Reuse canvas if dimensions match
            if (!canvas || canvas.width !== width || canvas.height !== height) {
                canvas = new OffscreenCanvas(width, height);
                ctx = canvas.getContext('2d');
            }
            ctx.drawImage(bitmap, 0, 0, width, height);
            bitmap.close();

            const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality });
            self.postMessage({ id, blob });
        } catch (err) {
            self.postMessage({ id, error: err.message });
        }
    };
`;

export class FrameEncoder {
    constructor() {
        this._useWorker = typeof OffscreenCanvas !== 'undefined' && typeof Worker !== 'undefined';
        this._worker = null;
        this._pending = new Map();
        this._nextId = 0;

        // Fallback canvas for main-thread encoding
        this._fallbackCanvas = null;
        this._fallbackCtx = null;

        if (this._useWorker) {
            try {
                const blob = new Blob([WORKER_CODE], { type: 'application/javascript' });
                const url = URL.createObjectURL(blob);
                this._worker = new Worker(url);
                URL.revokeObjectURL(url);

                this._worker.onmessage = (e) => {
                    const { id, blob, error } = e.data;
                    const resolver = this._pending.get(id);
                    if (resolver) {
                        this._pending.delete(id);
                        if (error) resolver.reject(new Error(error));
                        else resolver.resolve(blob);
                    }
                };
                this._worker.onerror = () => {
                    // Worker died — fall back to main thread
                    console.warn('[FrameEncoder] Worker died, falling back to main thread');
                    this._useWorker = false;
                    this._worker = null;
                };
                console.log('[FrameEncoder] Using OffscreenCanvas Worker (off-main-thread JPEG)');
            } catch (err) {
                console.warn('[FrameEncoder] Worker init failed, using main-thread fallback:', err.message);
                this._useWorker = false;
            }
        } else {
            console.log('[FrameEncoder] OffscreenCanvas not available, using main-thread canvas.toBlob()');
        }
    }

    /**
     * Encode a video frame to JPEG blob.
     * @param {HTMLVideoElement} video - Source video element
     * @param {number} width - Output width
     * @param {number} height - Output height
     * @param {number} quality - JPEG quality 0.0–1.0
     * @returns {Promise<Blob>} JPEG blob
     */
    async encode(video, width, height, quality) {
        if (this._useWorker && this._worker) {
            return this._encodeWorker(video, width, height, quality);
        }
        return this._encodeFallback(video, width, height, quality);
    }

    async _encodeWorker(video, width, height, quality) {
        // createImageBitmap() captures the video frame and transfers it to the worker
        // This is a near-zero-cost operation on the main thread
        const bitmap = await createImageBitmap(video, {
            resizeWidth: width,
            resizeHeight: height,
            resizeQuality: 'low', // 'low' is fastest for downscale
        });

        return new Promise((resolve, reject) => {
            const id = this._nextId++;
            this._pending.set(id, { resolve, reject });
            this._worker.postMessage(
                { id, bitmap, width, height, quality },
                [bitmap] // Transfer ownership — zero copy
            );
        });
    }

    _encodeFallback(video, width, height, quality) {
        if (!this._fallbackCanvas) {
            this._fallbackCanvas = document.createElement('canvas');
            this._fallbackCtx = this._fallbackCanvas.getContext('2d');
        }
        const canvas = this._fallbackCanvas;
        const ctx = this._fallbackCtx;
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        ctx.drawImage(video, 0, 0, width, height);
        return new Promise((resolve) => {
            canvas.toBlob((blob) => resolve(blob), 'image/jpeg', quality);
        });
    }

    dispose() {
        if (this._worker) {
            this._worker.terminate();
            this._worker = null;
        }
        this._pending.clear();
        this._fallbackCanvas = null;
        this._fallbackCtx = null;
    }
}

export default FrameEncoder;
