/**
 * Compute Delaunay triangulation from MediaPipe 468 face landmarks.
 * Uses delaunator for robust triangulation at runtime.
 * This avoids hardcoding ~1200 triangle indices and adapts to any face shape.
 */
import Delaunay from 'delaunator';

/**
 * Compute triangle indices from landmark array.
 * @param {Array<{x:number, y:number, z?:number}>} landmarks - 468 face landmarks
 * @returns {Array<[number, number, number]>} Array of [i, j, k] triangle index triples
 */
export function computeFaceMeshTriangles(landmarks) {
    if (!landmarks || landmarks.length < 10) return [];

    // Flatten to [x0, y0, x1, y1, ...] for delaunator
    const coords = new Float64Array(landmarks.length * 2);
    for (let i = 0; i < landmarks.length; i++) {
        coords[i * 2] = landmarks[i].x;
        coords[i * 2 + 1] = landmarks[i].y;
    }

    const delaunay = new Delaunay(coords);
    const tris = delaunay.triangles;
    const result = [];

    for (let i = 0; i < tris.length; i += 3) {
        const a = tris[i], b = tris[i + 1], c = tris[i + 2];

        // Filter out very large triangles (background/convex hull edges)
        const ax = landmarks[a].x, ay = landmarks[a].y;
        const bx = landmarks[b].x, by = landmarks[b].y;
        const cx = landmarks[c].x, cy = landmarks[c].y;

        // Compute triangle area (normalized coords, so threshold is small)
        const area = Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2;
        if (area < 0.005) {  // skip very large triangles that span the face edge
            result.push([a, b, c]);
        }
    }

    return result;
}

// Backward-compat: export empty array (real triangles computed at runtime)
export const FACE_MESH_TRIANGLES = [];
