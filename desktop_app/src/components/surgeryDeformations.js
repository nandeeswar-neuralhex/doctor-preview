/**
 * surgeryDeformations.js — 3D Face Surgery Deformation Engine
 *
 * Maps surgical procedures to MediaPipe landmark groups with directional
 * deformation vectors and smooth Gaussian falloff. Each procedure defines:
 *   - landmarks: affected indices
 *   - influence: neighboring indices pulled along (with falloff)
 *   - direction: [dx, dy, dz] unit deformation vector
 *   - maxMagnitude: max deformation in normalized coords
 *
 * Supports: Nose, Cheeks, Chin, Eyebrows, Eyes, Lips, Jawline, Forehead
 */

/* ── Landmark index groups by facial region ─────────────────────── */

const LANDMARK_GROUPS = {
    // Nose
    nose_bridge: [6, 197, 195, 5, 4, 45, 275],
    nose_tip: [1, 2, 98, 327, 168],
    nose_sides: [48, 115, 220, 45, 275, 440, 344, 278],
    nose_nostrils: [94, 141, 242, 97, 326, 462, 370, 324],
    nose_alar: [219, 218, 237, 44, 274, 457, 438, 439],
    nose_dorsum: [168, 6, 197, 195, 5, 4, 1, 19],

    // Cheeks
    left_cheek: [116, 117, 118, 119, 100, 36, 205, 187, 123, 147, 213, 192],
    right_cheek: [345, 346, 347, 348, 329, 266, 425, 411, 352, 376, 433, 416],
    left_cheekbone: [50, 101, 36, 205, 206, 207, 147],
    right_cheekbone: [280, 330, 266, 425, 426, 427, 376],

    // Chin
    chin_tip: [152, 175, 199, 200, 18, 313, 421, 396, 428],
    chin_sides: [150, 149, 148, 176, 140, 171, 379, 378, 377, 400, 369, 395],
    chin_lower: [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 148, 176],

    // Eyebrows
    left_brow: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    right_brow: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    left_brow_inner: [55, 65, 52, 53, 46],
    left_brow_outer: [70, 63, 105, 66, 107],
    right_brow_inner: [285, 295, 282, 283, 276],
    right_brow_outer: [300, 293, 334, 296, 336],

    // Eyes
    left_eye_upper: [159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154, 155, 157, 158],
    left_eye_lower: [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    right_eye_upper: [386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 384, 385],
    right_eye_lower: [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    left_under_eye: [111, 117, 118, 119, 120, 121, 128, 245, 193, 221],
    right_under_eye: [340, 346, 347, 348, 349, 350, 357, 465, 417, 441],

    // Lips
    upper_lip: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
    lower_lip: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    lip_corners: [61, 291],
    cupid_bow: [37, 0, 267, 39, 269, 40, 270],

    // Jawline
    left_jaw: [234, 93, 132, 58, 172, 136, 150, 149, 148, 176],
    right_jaw: [454, 323, 361, 288, 397, 365, 379, 378, 377, 400],
    jaw_angle_left: [172, 136, 150, 58, 132],
    jaw_angle_right: [395, 369, 379, 288, 361],

    // Forehead
    forehead_center: [10, 151, 9, 8, 107, 336, 108, 337, 69, 299],
    forehead_left: [21, 54, 103, 67, 109, 10, 338, 297],
    forehead_right: [251, 284, 332, 297, 338, 10, 109, 67],
};

/* ── Surgery Procedure Definitions ─────────────────────────────── */

export const SURGERY_CATEGORIES = {
    nose: {
        label: 'Nose (Rhinoplasty)',
        icon: '👃',
        color: '#a855f7',
        procedures: {
            nose_sharpen: {
                label: 'Sharpen Nose Bridge',
                description: 'Narrow the nasal bridge for a more defined look',
                groups: ['nose_bridge', 'nose_sides'],
                direction: [0, 0, 0.15],   // push forward (z)
                sideDir: [-0.12, 0, 0],     // compress sides inward (bilateral)
                bilateral: true,
                maxMag: 0.08,
                falloff: 0.6,
            },
            nose_tip_up: {
                label: 'Tip Rotation (Upward)',
                description: 'Rotate nasal tip upward for a refined profile',
                groups: ['nose_tip'],
                direction: [0, -0.15, 0.05],
                maxMag: 0.06,
                falloff: 0.5,
            },
            nose_tip_down: {
                label: 'Tip Rotation (Downward)',
                description: 'Lower the nasal tip for elongation',
                groups: ['nose_tip'],
                direction: [0, 0.12, -0.03],
                maxMag: 0.05,
                falloff: 0.5,
            },
            nose_narrow: {
                label: 'Narrow Nostrils',
                description: 'Alar reduction — narrow the nostril width',
                groups: ['nose_nostrils', 'nose_alar'],
                direction: [0, 0, 0],
                bilateral: true,
                sideDir: [-0.15, 0, 0],
                maxMag: 0.06,
                falloff: 0.4,
            },
            nose_bridge_reduce: {
                label: 'Reduce Dorsal Hump',
                description: 'Smooth the nasal bridge bump',
                groups: ['nose_dorsum'],
                direction: [0, 0, -0.12],
                maxMag: 0.05,
                falloff: 0.5,
            },
            nose_widen: {
                label: 'Widen Nose',
                description: 'Broaden the nasal bridge and nostrils',
                groups: ['nose_sides', 'nose_nostrils'],
                bilateral: true,
                sideDir: [0.12, 0, 0],
                direction: [0, 0, 0],
                maxMag: 0.05,
                falloff: 0.5,
            },
        },
    },
    cheeks: {
        label: 'Cheeks',
        icon: '😊',
        color: '#ec4899',
        procedures: {
            cheek_augment: {
                label: 'Cheek Augmentation',
                description: 'Add volume to cheekbones (filler/implant)',
                groups: ['left_cheekbone', 'right_cheekbone'],
                direction: [0, 0, 0.15],
                maxMag: 0.07,
                falloff: 0.7,
            },
            cheek_reduce: {
                label: 'Cheek Reduction',
                description: 'Buccal fat removal for slimmer cheeks',
                groups: ['left_cheek', 'right_cheek'],
                bilateral: true,
                sideDir: [-0.1, 0, -0.08],
                direction: [0, 0, 0],
                maxMag: 0.06,
                falloff: 0.6,
            },
            cheek_lift: {
                label: 'Cheek Lift',
                description: 'Lift sagging cheek tissue upward',
                groups: ['left_cheek', 'right_cheek'],
                direction: [0, -0.15, 0.05],
                maxMag: 0.06,
                falloff: 0.65,
            },
            cheek_contour: {
                label: 'Cheekbone Contour',
                description: 'Define cheekbone projection',
                groups: ['left_cheekbone', 'right_cheekbone'],
                direction: [0, -0.06, 0.12],
                maxMag: 0.06,
                falloff: 0.5,
            },
        },
    },
    chin: {
        label: 'Chin & Jaw',
        icon: '🗿',
        color: '#f97316',
        procedures: {
            chin_augment: {
                label: 'Chin Augmentation',
                description: 'Implant or filler for chin projection',
                groups: ['chin_tip'],
                direction: [0, 0.08, 0.15],
                maxMag: 0.08,
                falloff: 0.6,
            },
            chin_reduce: {
                label: 'Chin Reduction',
                description: 'Reduce chin projection',
                groups: ['chin_tip'],
                direction: [0, -0.05, -0.12],
                maxMag: 0.07,
                falloff: 0.6,
            },
            chin_sharpen: {
                label: 'V-Line Chin',
                description: 'Narrow the chin for a V-line shape',
                groups: ['chin_sides'],
                bilateral: true,
                sideDir: [-0.12, 0, 0],
                direction: [0, 0, 0],
                maxMag: 0.06,
                falloff: 0.6,
            },
            jaw_slim: {
                label: 'Jawline Slimming',
                description: 'Reduce jaw width (masseter reduction)',
                groups: ['left_jaw', 'right_jaw'],
                bilateral: true,
                sideDir: [-0.12, 0.04, 0],
                direction: [0, 0, 0],
                maxMag: 0.07,
                falloff: 0.65,
            },
            jaw_define: {
                label: 'Jawline Definition',
                description: 'Sharpen jaw angles',
                groups: ['jaw_angle_left', 'jaw_angle_right'],
                direction: [0, 0, 0.1],
                maxMag: 0.05,
                falloff: 0.4,
            },
        },
    },
    eyebrows: {
        label: 'Eyebrows',
        icon: '🤨',
        color: '#eab308',
        procedures: {
            brow_lift: {
                label: 'Brow Lift',
                description: 'Raise eyebrow position for an open look',
                groups: ['left_brow', 'right_brow'],
                direction: [0, -0.15, 0.02],
                maxMag: 0.06,
                falloff: 0.5,
            },
            brow_inner_lift: {
                label: 'Inner Brow Lift',
                description: 'Lift inner brow corners',
                groups: ['left_brow_inner', 'right_brow_inner'],
                direction: [0, -0.15, 0.03],
                maxMag: 0.05,
                falloff: 0.4,
            },
            brow_arch: {
                label: 'Brow Arch Enhance',
                description: 'Increase the arch for a dramatic look',
                groups: ['left_brow_outer', 'right_brow_outer'],
                direction: [0, -0.12, 0.04],
                maxMag: 0.05,
                falloff: 0.4,
            },
            brow_lower: {
                label: 'Lower Brows',
                description: 'Drop brow position for a softer look',
                groups: ['left_brow', 'right_brow'],
                direction: [0, 0.1, 0],
                maxMag: 0.04,
                falloff: 0.5,
            },
        },
    },
    eyes: {
        label: 'Eyes',
        icon: '👁️',
        color: '#3b82f6',
        procedures: {
            eye_open: {
                label: 'Eye Opening (Ptosis Fix)',
                description: 'Lift upper eyelids for wider eyes',
                groups: ['left_eye_upper', 'right_eye_upper'],
                direction: [0, -0.12, 0.02],
                maxMag: 0.04,
                falloff: 0.35,
            },
            eye_enlarge: {
                label: 'Eye Enlargement',
                description: 'Increase overall eye opening',
                groups: ['left_eye_upper', 'right_eye_upper', 'left_eye_lower', 'right_eye_lower'],
                direction: [0, -0.06, 0],  // upper goes up, lower goes down (handled in apply)
                maxMag: 0.04,
                falloff: 0.3,
            },
            under_eye_smooth: {
                label: 'Under Eye (Bag Removal)',
                description: 'Smooth under-eye bags and dark circles',
                groups: ['left_under_eye', 'right_under_eye'],
                direction: [0, -0.06, 0.08],
                maxMag: 0.04,
                falloff: 0.4,
            },
            cantho_lift: {
                label: 'Cat Eye / Canthoplasty',
                description: 'Lift outer eye corners upward',
                groups: ['left_eye_upper', 'right_eye_upper'],
                direction: [0, -0.1, 0],
                maxMag: 0.04,
                falloff: 0.3,
            },
        },
    },
    lips: {
        label: 'Lips',
        icon: '👄',
        color: '#ef4444',
        procedures: {
            lip_augment: {
                label: 'Lip Augmentation',
                description: 'Fuller lips (filler simulation)',
                groups: ['upper_lip', 'lower_lip'],
                direction: [0, 0, 0.12],
                maxMag: 0.05,
                falloff: 0.4,
            },
            lip_lift: {
                label: 'Lip Lift',
                description: 'Shorten philtrum, expose more upper lip',
                groups: ['upper_lip', 'cupid_bow'],
                direction: [0, -0.12, 0.05],
                maxMag: 0.04,
                falloff: 0.4,
            },
            lip_reduce: {
                label: 'Lip Reduction',
                description: 'Reduce lip volume',
                groups: ['upper_lip', 'lower_lip'],
                direction: [0, 0, -0.1],
                maxMag: 0.04,
                falloff: 0.4,
            },
            lip_corner_lift: {
                label: 'Corner Lip Lift',
                description: 'Lift downturned mouth corners',
                groups: ['lip_corners'],
                direction: [0, -0.15, 0.05],
                maxMag: 0.04,
                falloff: 0.3,
            },
        },
    },
    forehead: {
        label: 'Forehead',
        icon: '🧠',
        color: '#6366f1',
        procedures: {
            forehead_reduce: {
                label: 'Forehead Reduction',
                description: 'Lower the hairline / reduce forehead height',
                groups: ['forehead_center', 'forehead_left', 'forehead_right'],
                direction: [0, 0.1, 0],
                maxMag: 0.05,
                falloff: 0.6,
            },
            forehead_smooth: {
                label: 'Forehead Contouring',
                description: 'Smooth forehead bumps for a flat profile',
                groups: ['forehead_center'],
                direction: [0, 0, -0.1],
                maxMag: 0.04,
                falloff: 0.5,
            },
        },
    },
};

/* ── Deformation Engine ────────────────────────────────────────── */

/**
 * Apply a single surgery deformation to landmarks.
 *
 * @param {Array<{x,y,z}>} landmarks - original 468 3D landmarks
 * @param {string} procedureId - e.g. 'nose_sharpen'
 * @param {number} intensity - 0 to 1
 * @returns {Array<{x,y,z}>} new deformed landmarks
 */
export function applyDeformation(landmarks, procedureId, intensity) {
    if (!landmarks || intensity === 0) return landmarks;

    // Find procedure
    let procedure = null;
    for (const cat of Object.values(SURGERY_CATEGORIES)) {
        if (cat.procedures[procedureId]) {
            procedure = cat.procedures[procedureId];
            break;
        }
    }
    if (!procedure) return landmarks;

    // Collect all affected landmark indices
    const affectedSet = new Set();
    (procedure.groups || []).forEach(groupName => {
        (LANDMARK_GROUPS[groupName] || []).forEach(idx => affectedSet.add(idx));
    });
    const affected = [...affectedSet];

    // Compute centroid of affected landmarks
    let cx = 0, cy = 0;
    affected.forEach(i => { cx += landmarks[i].x; cy += landmarks[i].y; });
    cx /= affected.length;
    cy /= affected.length;

    // Find max distance from centroid (for falloff radius)
    let maxDist = 0;
    affected.forEach(i => {
        const dx = landmarks[i].x - cx;
        const dy = landmarks[i].y - cy;
        maxDist = Math.max(maxDist, Math.sqrt(dx * dx + dy * dy));
    });
    const radius = maxDist * (1 + procedure.falloff);

    // Deep clone landmarks
    const result = landmarks.map(l => ({ ...l }));

    // Apply deformation to all 468 landmarks with distance-based falloff
    const mag = procedure.maxMag * intensity;
    const [ddx, ddy, ddz] = procedure.direction;

    for (let i = 0; i < result.length; i++) {
        const dx = result[i].x - cx;
        const dy = result[i].y - cy;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > radius * 1.5) continue; // too far, no effect

        // Gaussian falloff
        const t = affectedSet.has(i) ? 1.0 : Math.exp(-(dist * dist) / (2 * radius * radius * procedure.falloff));
        if (t < 0.01) continue;

        const d = mag * t;

        // Apply main direction
        result[i].x += ddx * d;
        result[i].y += ddy * d;
        result[i].z += ddz * d;

        // Bilateral side deformation (nose narrowing, cheek reduction, etc.)
        if (procedure.bilateral && procedure.sideDir) {
            const [sx, sy, sz] = procedure.sideDir;
            // Left side: push right, Right side: push left
            const side = result[i].x < 0.5 ? 1 : -1;
            result[i].x += sx * d * side;
            result[i].y += sy * d;
            result[i].z += sz * d;
        }
    }

    return result;
}

/**
 * Apply multiple deformations in sequence.
 *
 * @param {Array<{x,y,z}>} landmarks - original 468 3D landmarks
 * @param {Object<string, number>} deformations - { procedureId: intensity, ... }
 * @returns {Array<{x,y,z}>} final deformed landmarks
 */
export function applyAllDeformations(landmarks, deformations) {
    if (!landmarks || !deformations) return landmarks;

    let result = landmarks;
    for (const [procId, intensity] of Object.entries(deformations)) {
        if (intensity > 0) {
            result = applyDeformation(result, procId, intensity);
        }
    }
    return result;
}

/**
 * Get all procedures as a flat list for UI.
 */
export function getAllProcedures() {
    const list = [];
    for (const [catId, cat] of Object.entries(SURGERY_CATEGORIES)) {
        for (const [procId, proc] of Object.entries(cat.procedures)) {
            list.push({
                id: procId,
                category: catId,
                categoryLabel: cat.label,
                categoryIcon: cat.icon,
                categoryColor: cat.color,
                ...proc,
            });
        }
    }
    return list;
}
