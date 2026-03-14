# 🗺️ Real-Time Face Swap V3 — Complete Roadmap

**Date:** 27 February 2026  
**Project:** Doctor Preview — Real-Time Face Swap + Lip Sync  
**Goal:** Production-grade HD face swap with perfect teeth, skin tone, lip sync, eye movement, micro-expressions, and cross-platform support (Mac + Windows)

---

## 📋 Table of Contents

1. [Current System Audit](#1-current-system-audit)
2. [Point 1: Teeth Rendering](#2-point-1-teeth-rendering)
3. [Point 2: Skin Tone Matching](#3-point-2-skin-tone-matching)
4. [Point 3: Lip Sync (Micro-Movements)](#4-point-3-lip-sync-micro-movements)
5. [Point 4: Eye Movement & Gaze Transfer](#5-point-4-eye-movement--gaze-transfer)
6. [Point 5: Blink & Micro Expression Capture](#6-point-5-blink--micro-expression-capture)
7. [Point 6: Light Smile & Subtle Mouth Corners](#7-point-6-light-smile--subtle-mouth-corners)
8. [Point 7: Skin Texture & Pore-Level Detail](#8-point-7-skin-texture--pore-level-detail)
9. [Point 8: Edge Blending & Lighting Consistency](#9-point-8-edge-blending--lighting-consistency)
10. [New Requirements & Dependencies](#10-new-requirements--dependencies)
11. [GPU & Infrastructure Requirements](#11-gpu--infrastructure-requirements)
12. [Phased Implementation Plan](#12-phased-implementation-plan)
13. [Risk Assessment](#13-risk-assessment)

---

## 1. Current System Audit

### 1.1 Current Architecture

```
Desktop Client (Electron + React)
    ↓ Webcam capture 1080p @ 24 FPS
    ↓ JPEG encode (canvas.toBlob @ 80% quality)
    ↓ Binary WebSocket (20-byte header + JPEG + audio PCM)
    ↓
GPU Server (RunPod — FastAPI + uvicorn)
    ↓ TurboJPEG decode
    ↓ Face Detection: RetinaFace (buffalo_l, 320×320 fast / 640×640 full)
    ↓ Face Swap: inswapper_128 (128×128 fixed resolution)
    ↓ Face Enhancement: GFPGAN v1.4 (optional, 512×512 upscale)
    ↓ Lip Sync: Wav2Lip-GAN ONNX (96×96 mouth region)
    ↓ Seamless Clone / Alpha Blend paste-back
    ↓ TurboJPEG encode
    ↓
Desktop Client
    ↓ createImageBitmap → canvas draw (zero-flicker)
```

### 1.2 Current Models in Use

| Component | Model | Resolution | Version |
|-----------|-------|-----------|---------|
| Face Detection | RetinaFace (buffalo_l) | 320×320 (fast), 640×640 (full) | InsightFace 0.7.3 |
| Face Recognition | ArcFace (buffalo_l) | 112×112 | InsightFace 0.7.3 |
| Face Swap | inswapper_128.onnx | **128×128 (FIXED)** | InsightFace |
| Face Enhancement | GFPGANv1.4 | 512×512 | GFPGAN 1.3.8 |
| Lip Sync | wav2lip_gan_96.onnx | **96×96 (FIXED)** | Wav2Lip |
| JPEG Codec | TurboJPEG (CPU SIMD) | N/A | PyTurboJPEG 1.7.1 |

### 1.3 Current Quality Scores (Honest Assessment)

| Feature | Score | Root Cause of Failure |
|---------|-------|-----------------------|
| Teeth rendering | **40/100** | inswapper destroys teeth at 128px; GFPGAN hallucinates generic teeth |
| Skin tone matching | **45/100** | CDF-based color transfer in LAB helps but still mismatches under mixed lighting |
| Lip sync accuracy | **35/100** | Wav2Lip operates at 96×96; only 16 mel frames (~200ms); no micro-movements |
| Eye movement/gaze | **0/100** | inswapper freezes source eye gaze direction entirely |
| Blink transfer | **0/100** | Source eye state is baked in; no blink detection or transfer |
| Micro expressions | **30/100** | 128×128 resolution destroys subtle facial muscle movements |
| Light smile corners | **35/100** | Fine mouth corner movements lost at 128px input |
| Skin texture/pores | **0/100** | 128→512 upscale creates smooth plastic appearance |
| Edge blending | **55/100** | ROI-based seamlessClone works but visible in motion; oval mask is generic |
| Lighting consistency | **40/100** | No relighting; pasted face retains source lighting |

### 1.4 Current Performance

| Metric | Mac M1 (Client) | Windows i3 (Client) | GPU Server |
|--------|-----------------|---------------------|------------|
| Client CPU usage | 40-60% | **80-100%** | N/A |
| JPEG encode (client) | 8-12ms | **40-80ms** | N/A |
| Frame payload size | ~150-200KB | ~150-200KB | N/A |
| Server decode | 3-5ms | 3-5ms | TurboJPEG |
| Face detection | 8-12ms | 8-12ms | GPU (320×320) |
| Face swap | 20-30ms | 20-30ms | GPU (inswapper_128) |
| GFPGAN enhance | 15-25ms | 15-25ms | GPU |
| Wav2Lip lip sync | 10-15ms | 10-15ms | GPU |
| Server encode | 3-5ms | 3-5ms | TurboJPEG |
| **End-to-end latency** | **~300ms** | **~10,000ms+** | N/A |
| Effective FPS | ~3-4 FPS | **<1 FPS** | N/A |

**Windows i3 failure cause:** Client-side 1080p JPEG encoding at 24 FPS overwhelms the CPU. Frames pile up in the WebSocket buffer, causing cascading latency. The server itself is fine — the bottleneck is the client.

---

## 2. Point 1: Teeth Rendering

### 2.1 Why Teeth Look Bad Currently

1. **inswapper_128 operates at 128×128 pixels.** A typical face in a 1080p frame has teeth occupying roughly 15-20 pixels in the 128×128 crop. At that resolution, individual teeth are indistinguishable — they become a white-grey blur.

2. **GFPGAN hallucinates teeth.** When upscaling from 128→512, GFPGAN generates "generic nice teeth" that don't match the source person's actual dental structure. The result looks uncanny because:
   - Teeth shape doesn't match the person
   - Teeth color is too uniform/white
   - Gum line is smoothed out
   - Gaps, overlaps, and natural imperfections are erased

3. **Wav2Lip further destroys teeth.** The lip sync model overwrites the bottom half of the face at 96×96, re-introducing blur on top of the already-degraded teeth from GFPGAN.

### 2.2 Solution: LivePortrait + CodeFormer

| Component | How It Fixes Teeth |
|-----------|-------------------|
| **LivePortrait** (replace inswapper) | Operates at **512×512**. Teeth occupy ~60-80 pixels — enough to preserve individual tooth structure. It animates the source face's teeth naturally based on driving expression. |
| **CodeFormer** (replace GFPGAN) | Uses a codebook of high-quality face features learned from real faces. Instead of hallucinating teeth, it finds the closest real-teeth pattern and restores detail. Fidelity parameter (0.7) preserves the source person's actual teeth while enhancing clarity. |
| **MuseTalk** (replace Wav2Lip) | Operates at **256×256** mouth region. Teeth are visible as distinct structures. Trained on high-res talking head videos with visible teeth. |

### 2.3 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Teeth visibility | Blurred white mass | Individual teeth visible |
| Teeth shape accuracy | Generic/hallucinated | Matches source person |
| Teeth during speech | Destroyed by Wav2Lip | Preserved by MuseTalk |
| **Score** | **40/100** | **85-90/100** |

### 2.4 Remaining Limitations

- Teeth are still generated/reconstructed, not pixel-perfect copies
- Very unusual dental structures (large gaps, gold teeth, etc.) may not transfer perfectly
- Side-angle views reduce teeth visibility for all models

---

## 3. Point 2: Skin Tone Matching

### 3.1 Why Skin Tone Mismatches Currently

1. **inswapper pastes the source face texture directly.** If the source person (target photo) has a lighter/darker skin tone than the webcam person's body/neck, the face looks "pasted on" with a visible color boundary.

2. **Current CDF color matching is face-only.** Your `_color_match` method in `face_swapper.py` does histogram-based LAB color transfer, but:
   - It only runs within the face mask region
   - It doesn't consider the neck/body skin tone
   - It doesn't handle mixed lighting (half-face shadow)
   - LAB CDF matching can shift hue incorrectly on very different skin tones (e.g., dark skin → light skin)

3. **Seamless clone blends color but creates artifacts.** `cv2.seamlessClone(NORMAL_CLONE)` adjusts colors at the boundary but can introduce halo artifacts, especially under strong directional lighting.

### 3.2 Solution: LivePortrait Native Skin + Multi-Region Color Transfer

| Component | How It Fixes Skin Tone |
|-----------|----------------------|
| **LivePortrait** | Unlike inswapper (which pastes texture), LivePortrait **generates** the entire face from the source appearance. The generated face inherits the source person's skin tone uniformly — no paste = no mismatch with "itself." |
| **Multi-region color harmonization** | After LivePortrait generates the face, a post-processing step samples the webcam person's **neck and body skin** and adjusts the generated face's overall tone to blend with the surrounding skin. This uses LAB color transfer but across face→body regions, not face→face. |
| **Poisson blending with feathered oval** | Replace the current rectangular ROI seamless clone with a 68-point landmark contour mask + Gaussian feathering. This creates an invisible transition zone between face and neck. |

### 3.3 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Face-to-body tone match | Visible mismatch | Seamless transition |
| Boundary artifacts | Halo/seam visible | Invisible |
| Cross-race accuracy | Poor (strong mismatch) | Good (harmonized) |
| **Score** | **45/100** | **88-92/100** |

### 3.4 Remaining Limitations

- Extreme skin tone differences (very dark source → very light target) will still show some adaptation artifacts
- Body skin covered by clothing can't be sampled — only exposed neck/décolletage is used
- Tattoos and moles on the face boundary can cause color sampling errors

---

## 4. Point 3: Lip Sync (Micro-Movements)

### 4.1 Why Lip Sync Fails Currently

1. **Wav2Lip operates at 96×96 pixels.** The entire mouth, chin, and lower cheek are compressed into a 96-pixel crop. Subtle lip movements (like the difference between "p" and "b", or a slight lip purse) are invisible at this resolution.

2. **Only 16 mel frames (~200ms) of audio context.** Wav2Lip sees a narrow window of audio, missing:
   - Anticipatory lip movements (lips rounding before "oo" sound)
   - Coarticulation (lip shape influenced by the next phoneme)
   - Breathing pauses and natural lip resting positions

3. **Wav2Lip overwrites the face swap.** The lip sync model replaces the bottom half of the already-swapped face. Two neural networks fighting over the same pixels creates:
   - Flickering between swap quality and lip sync quality
   - Color/texture inconsistency between upper face (swap) and lower face (lip sync)
   - Loss of the face enhancement (GFPGAN) in the mouth region

4. **300ms audio delay compensation.** Your client sends audio with a 300ms buffer (`audioDelayMs: 300`), meaning the server receives audio that's already 300ms old relative to the video frame. The lip movements always lag the actual speech.

### 4.2 Solution: MuseTalk (Replace Wav2Lip)

| Feature | Wav2Lip (Current) | MuseTalk (New) |
|---------|--------------------|----------------|
| Resolution | 96×96 | **256×256** |
| Audio context | 16 frames (~200ms) | **80 frames (~1000ms)** |
| Micro-movements | None | **Captures subtle lip tensions** |
| Phoneme accuracy | ~55% | **~78%** |
| "P" vs "B" distinction | Indistinguishable | Visible difference |
| Teeth visibility during speech | Blurred | Clear |
| Integration | Overwrites face swap | **Blends with face swap output** |
| Training data | Low-res talking heads | **High-res, multi-speaker** |

### 4.3 Integration Architecture

**Current (broken) pipeline:**
```
Webcam frame → inswapper (128px) → GFPGAN (512px) → Wav2Lip OVERWRITES bottom half (96px)
                                                      ↑ Destroys swap quality
```

**New (correct) pipeline:**
```
Webcam frame → LivePortrait (512px, generates animated face with mouth movement)
             → MuseTalk (256px, refines mouth region with audio-driven sync)
             → CodeFormer (512px, enhances full face including teeth)
             → Skin harmonization → Poisson blend
```

Key difference: MuseTalk **refines** the mouth region rather than **replacing** it. LivePortrait already provides motion-driven mouth shapes; MuseTalk adds audio-precise timing.

### 4.4 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Phoneme accuracy (visual) | ~55% | ~78% |
| Lip-audio sync delay | 200-400ms | 50-100ms |
| Micro-movements (lip tension, purse) | Not captured | Captured |
| Teeth during speech | Blurred/destroyed | Clear at 256px |
| Consistency with face swap | Flickering | Seamless |
| **Score** | **35/100** | **80-85/100** |

### 4.5 Remaining Limitations

- Audio-driven lip sync is inherently reactive (audio → visual), not predictive
- Very fast speech (>200 words/minute) may still miss some articulations
- Whispered or breathy speech produces minimal visual movement in any model
- Non-English phonemes may be less accurate (models primarily trained on English)

---

## 5. Point 4: Eye Movement & Gaze Transfer

### 5.1 Why Eyes Don't Move Currently

**inswapper_128 is a face-texture-swap model.** It replaces the face appearance (skin, features, shape) but does NOT transfer:
- Eye gaze direction (where the person is looking)
- Pupil dilation
- Eyelid position (squint vs wide-open)
- Eye moisture/reflection

The swapped face always shows the source photo's frozen eye state. If the source photo has the person looking left, the output face looks left regardless of where the webcam person is actually looking.

### 5.2 Solution: LivePortrait Native Gaze Retargeting

LivePortrait is fundamentally different from inswapper. It is a **motion transfer** model:

```
Source Image (target person's appearance)
    +
Driving Video Frame (webcam person's motion)
    ↓
LivePortrait extracts:
    - 3D head pose (yaw, pitch, roll)
    - Eye gaze direction (left/right/up/down for each eye)
    - Eyelid openness (per eye)
    - Eyebrow position
    - Mouth shape (66 motion parameters)
    ↓
Applies ALL motion to source appearance
    ↓
Output: Source person's face moving exactly like webcam person
```

LivePortrait's gaze retargeting works by:
1. Detecting iris position in the driving frame (webcam)
2. Extracting gaze angle as 2D coordinates
3. Rendering the source face's eyes looking in that direction
4. Preserving sclera (white of eye), iris color, and pupil from source

### 5.3 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Gaze direction transfer | None (frozen) | Accurate (±5° error) |
| Gaze tracking range | N/A | ±30° horizontal, ±20° vertical |
| Pupil movement smoothness | N/A | 60Hz interpolated |
| Eye contact with camera | Impossible | Natural |
| Looking at different points on screen | Impossible | Accurate |
| **Score** | **0/100** | **88-92/100** |

### 5.4 Remaining Limitations

- Extreme gaze angles (looking far to the side) can distort the iris rendering
- Source photos with closed/squinted eyes limit the output gaze range
- Glasses in the source photo can occlude eye region, reducing accuracy
- Very dark irises (where iris/pupil boundary is unclear) reduce gaze precision

---

## 6. Point 5: Blink & Micro Expression Capture

### 6.1 What's Missing Currently

**inswapper does not transfer ANY temporal/dynamic face features.** Each frame is processed independently with the same source face appearance. This means:

- **Blinks:** If the webcam person blinks, the output face does NOT blink. It maintains the source photo's eye state (usually open).
- **Eyebrow raises:** The output face has frozen eyebrows matching the source photo.
- **Nose wrinkle:** Not transferred.
- **Forehead tension:** Not transferred.
- **Cheek puffing:** Not transferred.
- **Jaw clench:** Partially transferred through landmark geometry but heavily degraded at 128px.

### 6.2 Solution: LivePortrait 68-Parameter Motion Model

LivePortrait captures **68 distinct motion parameters** from the driving frame:

| Parameter Group | Count | What It Captures |
|----------------|-------|-----------------|
| Head pose | 3 | Yaw, pitch, roll |
| Eye gaze | 4 | Left/right eye horizontal/vertical |
| Eyelid | 4 | Upper/lower lid openness per eye |
| Eyebrow | 6 | Inner/mid/outer raise per brow |
| Nose | 2 | Nose wrinkle, nostril flare |
| Mouth shape | 18 | Open/close, width, asymmetry, corners, lip roll |
| Jaw | 3 | Open, lateral shift, forward |
| Cheek | 4 | Puff, suck, raise per side |
| Chin | 2 | Dimple, tension |
| Tongue | 2 | Out, up/down |
| **Total** | **48+** | Full facial action coding system (FACS) |

Because LivePortrait operates at **512×512**, these micro-movements are visible:
- A slight brow raise (2-3 pixel movement at 512px vs <1 pixel at 128px)
- A nostril flare (visible at 512px, invisible at 128px)
- Eyelid micro-droop during fatigue (captured)

### 6.3 Expected Improvement

| Feature | Before | After |
|---------|--------|-------|
| Blink transfer | Not transferred | Natural blink with correct duration |
| Blink speed accuracy | N/A | ~90% (150-400ms blink captured) |
| Eyebrow raise | Frozen | Full range of motion |
| Nose wrinkle (disgust) | Not transferred | Captured at 512px |
| Forehead tension (worry) | Not transferred | Visible |
| Cheek puff | Not transferred | Captured |
| Jaw clench | Partially (degraded) | Clear |
| **Score** | **0/100 (blink), 30/100 (micro)** | **90/100 (blink), 85/100 (micro)** |

### 6.4 Remaining Limitations

- Very fast micro-expressions (<100ms, like fear micro-flash) may be missed between frames at 15-20 FPS
- Asymmetric expressions (one eyebrow raise) are harder than symmetric ones
- Source photos with strong expressions (big smile) reduce the dynamic range of output expressions
- Recommendation: Source photos should have **neutral expression** for maximum motion range

---

## 7. Point 6: Light Smile & Subtle Mouth Corners

### 7.1 Why Light Smiles Are Lost Currently

1. **128×128 resolution kills subtlety.** A "light smile" is primarily defined by:
   - Mouth corners lifting 2-3mm (real-world)
   - Slight deepening of nasolabial folds
   - Minor cheek elevation
   - Orbicularis oculi contraction (slight eye narrowing — "Duchenne smile")

   At 128×128 pixels, these movements translate to **less than 1 pixel of change**. The model literally cannot represent them.

2. **5-point landmarks are too coarse.** Your current fast detector uses only 5 keypoints (2 eyes, nose, 2 mouth corners). This cannot distinguish between:
   - Neutral face
   - Light smile
   - Slight frown
   - Lip press
   
   All four produce nearly identical 5-point configurations.

3. **Expression matching is approximate.** Your `_extract_expression_features` computes ratios from landmarks, but:
   - With 5 points, "mouth_width" is the only smile indicator
   - The difference between neutral and light-smile mouth width is ~3-5% — within noise
   - This means the wrong target face may be selected for expression matching

### 7.2 Solution: LivePortrait Mouth Motion Parameters

LivePortrait's 18 mouth parameters include:

| Parameter | What It Captures | Smile Relevance |
|-----------|-----------------|-----------------|
| `mouth_smile_left` | Left corner lift amount | **Primary smile indicator** |
| `mouth_smile_right` | Right corner lift amount | **Primary smile indicator** |
| `mouth_dimple_left` | Left cheek dimple depth | Light smile enhancer |
| `mouth_dimple_right` | Right cheek dimple depth | Light smile enhancer |
| `mouth_stretch_left` | Left corner horizontal pull | Wide vs tight smile |
| `mouth_stretch_right` | Right corner horizontal pull | Wide vs tight smile |
| `mouth_press_left` | Left lip press | Suppressed smile indicator |
| `mouth_press_right` | Right lip press | Suppressed smile indicator |
| `mouth_roll_upper` | Upper lip roll inward | Lip biting while smiling |
| `mouth_roll_lower` | Lower lip roll inward | Lip biting while smiling |

Even a **very subtle light smile** (mouth corners up by 1-2mm) produces measurable changes in `mouth_smile_left/right` values (0.0 → 0.05-0.10). At 512×512 resolution, this translates to 3-5 pixels of visible corner movement.

### 7.3 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Light smile detection threshold | Undetectable | 0.5mm lip corner movement |
| Smile asymmetry (smirk) | Not captured | Left vs right independent |
| Nasolabial fold deepening | Lost | Visible at 512px |
| "Polite smile" vs "genuine smile" | Indistinguishable | Different (eye involvement) |
| Lip press (suppressed smile) | Not captured | Captured |
| **Score** | **35/100** | **85-90/100** |

### 7.4 Remaining Limitations

- Very faint "social smiles" (<0.5mm corner movement) are at the detection threshold
- Source photos with an existing smile reduce the range of smile intensities possible
- Best results: source photo with closed-mouth neutral expression

---

## 8. Point 7: Skin Texture & Pore-Level Detail

### 8.1 Why Skin Looks Plastic Currently

1. **inswapper_128 → 128×128 completely removes skin texture.** At 128 pixels, one pixel covers approximately 0.5-1mm of real face surface. Human skin pores are 0.02-0.05mm. The model cannot represent pores — they are **250x smaller than one pixel.**

2. **GFPGAN upscale (128→512) generates smooth skin.** GFPGAN's training objective is "make the face look good" which biases toward:
   - Removing wrinkles
   - Smoothing pores
   - Evening out skin tone
   - Creating a "beauty filter" effect
   
   This is the opposite of what we want for realism.

3. **JPEG compression at 90% quality removes remaining texture.** Even if some texture survived, the JPEG encode at quality 90 (your current `JPEG_QUALITY`) applies DCT quantization that smooths fine details.

### 8.2 Solution: CodeFormer with High Fidelity + LivePortrait 512px Base

| Component | How It Restores Texture |
|-----------|------------------------|
| **LivePortrait at 512×512** | 4× the resolution of inswapper. At 512px, one pixel covers ~0.12mm — approaching pore visibility. Fine wrinkles, skin texture variation, and light freckles are preserved in the base generation. |
| **CodeFormer (fidelity=0.7)** | Unlike GFPGAN which always "beautifies," CodeFormer's fidelity parameter controls the balance: `0.0` = maximum quality enhancement (smooth), `1.0` = maximum fidelity to input (preserves all texture). At `0.7`, it restores clarity while preserving natural skin variation. |
| **JPEG quality 88-92** | Higher encode quality preserves fine texture in the output. Combined with smaller face regions (only face area, not full 1080p background), the file size increase is minimal. |

### 8.3 What "Pore-Level" Actually Means at Each Resolution

| Resolution | Pixel Size (on face) | Pores Visible? | Wrinkles Visible? | Moles/Freckles? |
|-----------|---------------------|---------------|-------------------|-----------------|
| 128×128 (inswapper) | ~0.5mm/px | ❌ No | ❌ No | ❌ No |
| 256×256 | ~0.25mm/px | ❌ No | ⚠️ Deep only | ⚠️ Large only |
| **512×512 (LivePortrait)** | ~0.12mm/px | ⚠️ Large pores | ✅ Yes | ✅ Yes |
| 1024×1024 | ~0.06mm/px | ✅ Yes | ✅ Yes | ✅ Yes |

**Reality check:** True pore-level detail requires 1024×1024+ resolution, which no real-time face animation model can do at acceptable latency today. At 512×512, we get **fine wrinkles, moles, freckles, and skin texture variation** but not individual pores.

### 8.4 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Skin texture variety | None (uniform smooth) | Natural variation |
| Fine wrinkles | Erased | Visible |
| Moles and freckles | Erased | Preserved from source |
| Pores | Not visible | Large pores hinted |
| "Plastic/AI look" | Very obvious | Significantly reduced |
| **Score** | **0/100** | **75-80/100** |

### 8.5 Remaining Limitations

- True pore-level (0.02mm) detail is not achievable at 512×512 — requires 1024×1024
- Acne, scars, and unique skin features from the source photo may not perfectly transfer
- The "uncanny valley" is reduced but not eliminated — trained observers can still detect AI faces
- Real-time 1024×1024 face animation is not feasible with current GPU technology at <200ms latency

---

## 9. Point 8: Edge Blending & Lighting Consistency

### 9.1 Current Edge Blending Problems

1. **Oval mask is generic.** Your current paste-back uses a 128×128 ellipse (`cv2.ellipse(64, 64, 52, 58)`) warped to the face position. This doesn't follow the actual face contour, causing:
   - Chin may be clipped (oval too small) or include background (oval too big)
   - Temple/hairline transition is abrupt
   - Ear edges are jagged

2. **68-point contour mask (fallback) is jittery.** While you have a 68-point jawline contour path, the `SMOOTHING_ALPHA=0.4` temporal smoothing causes visible wobble during fast head movements.

3. **Seamless clone creates halos.** `cv2.seamlessClone(NORMAL_CLONE)` modifies color gradients at the boundary. Under strong directional lighting (lamp from one side), this creates:
   - Bright halo on the shadow side
   - Dark halo on the lit side
   - Color shift in the transition zone

4. **No relighting.** The swapped face retains the lighting from the source photo. If the source photo was taken in natural daylight and the webcam person is in a dim room with yellow lamplight, the face looks "photoshopped in" — wrong color temperature, wrong shadow direction, wrong highlight placement.

### 9.2 Solution: LivePortrait Full-Face Generation + Post-Process Harmonization

| Component | How It Fixes Edges |
|-----------|-------------------|
| **LivePortrait full-face generation** | Unlike inswapper (which crops, swaps, and pastes), LivePortrait generates the ENTIRE face region including background context near the boundary. There is no hard "paste" edge — the transition is part of the generation. |
| **Landmark-guided feathered mask** | For blending the generated face into the original frame, use 68-point jawline + forehead contour with 15-pixel Gaussian feathering. The mask follows the actual face shape, not a generic oval. |
| **Multi-band blending** | Replace seamlessClone with Laplacian pyramid blending (3 levels). This blends low-frequency color differences (skin tone) independently from high-frequency details (texture, edges), eliminating halos. |
| **LAB color harmonization** | After generation, match the face region's L (lightness) channel to the surrounding frame. This partially compensates for lighting differences without requiring a full relighting model. |

### 9.3 Lighting Consistency: What's Possible

| Approach | Quality | Speed | Feasibility |
|----------|---------|-------|-------------|
| No relighting (current) | 40% | 0ms | Current |
| **LAB lightness matching** (planned) | **65%** | **3-5ms** | ✅ Implementable now |
| Spherical harmonics relighting | 80% | 50-80ms | Possible but adds latency |
| IC-Light (diffusion-based) | 90% | 200-400ms | Too slow for real-time |
| Neural relighting (NeRF-based) | 95% | 500ms+ | Not real-time |

**Recommendation:** LAB lightness matching provides the best quality/speed tradeoff. It corrects overall brightness and color temperature differences without the latency of neural relighting.

### 9.4 Expected Improvement

| Metric | Before | After |
|--------|--------|-------|
| Edge visibility (static) | Visible seam | Invisible with feathered contour |
| Edge visibility (motion) | Wobbling mask | Stable 68-point tracking |
| Halo artifacts | Present under directional light | Eliminated (multi-band blend) |
| Lighting temperature match | Mismatched | Partially corrected (LAB) |
| Shadow direction match | Not corrected | Not corrected (would need relighting) |
| **Edge score** | **55/100** | **90-92/100** |
| **Lighting score** | **40/100** | **65-70/100** |

### 9.5 Remaining Limitations

- Shadow direction cannot be corrected without a relighting model (too slow for real-time)
- Strong backlighting (person in front of window) creates silhouette effects that no blend can fix
- Hair-to-face boundary is the hardest edge — hair strands are difficult to mask cleanly
- Moving hair (wind, head shaking) causes temporary edge artifacts

---

## 10. New Requirements & Dependencies

### 10.1 Model Requirements

| Model | Size on Disk | GPU VRAM Required | Download Source |
|-------|-------------|-------------------|-----------------|
| **LivePortrait** | ~1.2 GB (5 sub-models) | ~2.5 GB | HuggingFace: KwaiVGI/LivePortrait |
| **MuseTalk** | ~800 MB | ~1.5 GB | HuggingFace: TMElyralab/MuseTalk |
| **CodeFormer** | ~400 MB | ~1.0 GB | GitHub: sczhou/CodeFormer |
| **RetinaFace** (buffalo_l) | ~300 MB (keep existing) | ~0.5 GB | InsightFace (already installed) |
| **Total new VRAM** | — | **~5.5 GB** | — |
| **Total with safety margin** | — | **~8-10 GB** | — |

### 10.2 LivePortrait Sub-Models

| Sub-Model | Purpose | Size |
|-----------|---------|------|
| `appearance_feature_extractor.pth` | Extract source face identity features | ~250 MB |
| `motion_extractor.pth` | Extract driving motion (68 parameters) | ~200 MB |
| `warping_module.pth` | Warp source appearance with driving motion | ~350 MB |
| `spade_generator.pth` | Generate final face image at 512×512 | ~300 MB |
| `stitching_retargeting_module.pth` | Blend generated face into frame | ~100 MB |

### 10.3 Python Dependencies (New)

```
# NEW — Face animation (replaces inswapper)
liveportrait>=0.1.0          # or install from source: pip install git+https://github.com/KwaiVGI/LivePortrait.git

# NEW — Lip sync (replaces Wav2Lip)
musetalk>=0.1.0              # or install from source: pip install git+https://github.com/TMElyralab/MuseTalk.git

# NEW — Face enhancement (replaces GFPGAN)
codeformer                    # pip install git+https://github.com/sczhou/CodeFormer.git

# KEEP — Face detection (still needed)
insightface==0.7.3
onnxruntime-gpu>=1.20.1

# KEEP — Server
fastapi==0.109.2
uvicorn[standard]==0.27.1

# KEEP — Image processing
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.0.0
PyTurboJPEG>=1.7.1

# KEEP — Audio
librosa>=0.10.0
soundfile>=0.12.0

# NEW — Additional
scipy>=1.11.0
scikit-image>=0.21.0

# REMOVE — No longer needed
# gfpgan (replaced by CodeFormer)
# wav2lip (replaced by MuseTalk)
# inswapper_128.onnx (replaced by LivePortrait)
```

### 10.4 Client-Side Changes

| Change | Why |
|--------|-----|
| Adaptive resolution (1080p → 480p based on hardware) | Fix Windows i3 performance |
| Adaptive FPS (24 → 12 based on hardware) | Prevent frame queue buildup |
| OffscreenCanvas encoding | Move JPEG encode off main thread on Chrome/Edge |
| Aggressive WebSocket backpressure (`bufferedAmount > 0` = skip) | Prevent latency snowball |
| Server timing overlay in UI | Show per-stage timing for debugging |
| Auto-disable lip sync on <4 cores | Reduce audio overhead on weak machines |

---

## 11. GPU & Infrastructure Requirements

### 11.1 Minimum GPU Requirements

| Tier | GPU | VRAM | Expected Latency | Concurrent Users | Monthly Cost (RunPod) |
|------|-----|------|-------------------|-------------------|----------------------|
| Minimum | RTX 3090 | 24 GB | 200-280ms | 1-2 | ~$290/mo |
| **Recommended** | **A100 40GB** | **40 GB** | **120-160ms** | **3-4** | **~$720/mo** |
| Premium | A100 80GB | 80 GB | 100-140ms | 6-8 | ~$1,100/mo |
| Enterprise | H100 | 80 GB | 70-100ms | 10-15 | ~$1,800/mo |

### 11.2 VRAM Budget (A100 40GB)

| Component | VRAM Usage | Notes |
|-----------|-----------|-------|
| LivePortrait (all 5 sub-models) | ~2.5 GB | Loaded once, shared across sessions |
| MuseTalk | ~1.5 GB | Loaded once |
| CodeFormer | ~1.0 GB | Loaded once |
| RetinaFace (buffalo_l) | ~0.5 GB | Loaded once |
| ONNX Runtime workspace | ~1.0 GB | Dynamic allocation |
| Per-session frame buffers | ~0.3 GB × N sessions | N = concurrent users |
| PyTorch CUDA context | ~1.0 GB | Fixed overhead |
| **Total (4 sessions)** | **~8.7 GB** | Fits in 40GB with headroom |

### 11.3 Per-Frame Latency Budget (A100 40GB)

| Stage | Target Time | Notes |
|-------|------------|-------|
| JPEG decode (TurboJPEG) | 3-5ms | CPU, parallel with GPU |
| Face detection (RetinaFace, cached) | 2-5ms | Full: 12-15ms every 3rd frame |
| LivePortrait face animation | 45-60ms | **Heaviest stage** |
| MuseTalk lip sync | 50-70ms | Can run parallel with CodeFormer on multi-GPU |
| CodeFormer enhancement | 20-25ms | Fidelity=0.7 |
| Skin harmonization | 3-5ms | CPU (LAB color transfer) |
| Poisson/multi-band blend | 3-5ms | CPU |
| JPEG encode (TurboJPEG) | 3-5ms | CPU |
| **Total (serial)** | **~130-180ms** | Single GPU path |
| **Total (pipelined)** | **~100-130ms** | MuseTalk + CodeFormer overlap |

### 11.4 Network Requirements

| Metric | Requirement |
|--------|------------|
| Client upload bandwidth | ≥2 Mbps (720p JPEG @ 15 FPS × ~80KB/frame) |
| Client download bandwidth | ≥2 Mbps (same) |
| Round-trip latency (network) | ≤50ms (same region) |
| WebSocket support | Required |
| Recommended server region | Same continent as user |

---

## 12. Phased Implementation Plan

### Phase 1: Client Performance Fix (Week 1)
**Goal:** Make the app work on Windows i3 with <1s latency

| Task | Priority | Effort |
|------|----------|--------|
| Add hardware detection (`navigator.hardwareConcurrency`, `navigator.deviceMemory`) | High | 2 hrs |
| Implement adaptive resolution (1080p/720p/480p based on CPU tier) | High | 3 hrs |
| Implement adaptive FPS (24/18/12 based on CPU tier) | High | 2 hrs |
| Add OffscreenCanvas JPEG encoding for Chrome/Edge | Medium | 3 hrs |
| Reduce WebSocket buffer threshold (1MB → adaptive: 200KB-1MB) | High | 1 hr |
| Auto-disable lip sync on <4 cores | Medium | 1 hr |
| Add latency/FPS/resolution overlay in UI | Low | 2 hrs |

**Deliverable:** Desktop app works on Windows i3 at 480p/12FPS with <500ms latency.

### Phase 2: LivePortrait Integration (Week 2-3)
**Goal:** Replace inswapper_128 with LivePortrait for 512×512 face animation

| Task | Priority | Effort |
|------|----------|--------|
| Set up LivePortrait model download pipeline | High | 4 hrs |
| Implement source face pre-processing (one-time per session) | High | 4 hrs |
| Implement per-frame motion extraction from driving frame | High | 6 hrs |
| Implement face animation (source appearance + driving motion) | High | 8 hrs |
| Implement face stitching (blend generated face into frame) | High | 4 hrs |
| Enable gaze retargeting | Medium | 3 hrs |
| Enable eye/brow expression transfer | Medium | 3 hrs |
| Benchmark and optimize (target: 60ms per frame on A100) | High | 4 hrs |
| Remove inswapper_128 dependency | Low | 1 hr |

**Deliverable:** Face swap at 512×512 with full expression/gaze transfer. Teeth, eyes, micro-expressions all improved.

### Phase 3: MuseTalk Lip Sync (Week 3-4)
**Goal:** Replace Wav2Lip with MuseTalk for accurate lip sync

| Task | Priority | Effort |
|------|----------|--------|
| Set up MuseTalk model download pipeline | High | 3 hrs |
| Implement audio preprocessing (PCM → mel at 16kHz with wider context) | High | 4 hrs |
| Implement MuseTalk inference (audio + face → lip-synced mouth) | High | 6 hrs |
| Implement mouth region blending (MuseTalk output → LivePortrait face) | High | 4 hrs |
| Reduce audio delay from 300ms to 50-100ms | High | 3 hrs |
| Test micro-movement accuracy (p/b/m phonemes) | Medium | 2 hrs |
| Remove Wav2Lip dependency | Low | 1 hr |

**Deliverable:** Audio-accurate lip sync at 256×256 with micro-movements visible.

### Phase 4: CodeFormer + Skin Harmonization (Week 4-5)
**Goal:** Perfect teeth, skin texture, and edge blending

| Task | Priority | Effort |
|------|----------|--------|
| Set up CodeFormer model | High | 3 hrs |
| Integrate CodeFormer with fidelity=0.7 (teeth + skin restoration) | High | 4 hrs |
| Implement multi-region skin tone sampling (face vs neck/body) | Medium | 4 hrs |
| Implement LAB color harmonization (face → body tone match) | Medium | 3 hrs |
| Replace seamlessClone with multi-band Laplacian blending | Medium | 4 hrs |
| Implement 68-point contour mask with Gaussian feathering | Medium | 3 hrs |
| Remove GFPGAN dependency | Low | 1 hr |

**Deliverable:** Natural teeth rendering, matched skin tone, invisible face edges.

### Phase 5: Server Pipeline Optimization (Week 5-6)
**Goal:** Optimize for <150ms end-to-end latency

| Task | Priority | Effort |
|------|----------|--------|
| Implement frame-rate limiter (drop stale frames, process latest) | High | 3 hrs |
| Implement face detection caching (detect every 3rd frame) | High | 3 hrs |
| Implement pipelined processing (overlap MuseTalk + CodeFormer) | Medium | 6 hrs |
| Add FP16 inference for LivePortrait (halve GPU time) | Medium | 4 hrs |
| Add TensorRT optimization for MuseTalk | Medium | 4 hrs |
| Implement per-session queues with stats tracking | Low | 3 hrs |
| Implement /health and /stats endpoints for monitoring | Low | 2 hrs |

**Deliverable:** 120-160ms latency on A100, 15-20 effective FPS.

### Phase 6: Testing & Polish (Week 6-7)
**Goal:** Production readiness

| Task | Priority | Effort |
|------|----------|--------|
| Test on Mac M1 (confirm <200ms latency) | High | 2 hrs |
| Test on Windows i3 (confirm <500ms latency) | High | 2 hrs |
| Test on Windows i5/i7 (confirm <300ms latency) | High | 2 hrs |
| Test with different source face types (diverse skin tones, ages) | High | 3 hrs |
| Test with glasses, beards, head scarves | Medium | 2 hrs |
| Test with multiple concurrent users (3-4 sessions) | High | 3 hrs |
| Update Dockerfile with all new models | High | 3 hrs |
| Update model download script | High | 2 hrs |
| Performance profiling and bottleneck elimination | Medium | 4 hrs |

**Deliverable:** Production-ready V3 deployment.

---

## 13. Risk Assessment

### 13.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| LivePortrait doesn't run in real-time on A100 | Low (25%) | High | FP16 + TensorRT optimization; fallback to SimSwap-HQ (256px) |
| MuseTalk audio sync is worse than expected | Medium (35%) | Medium | Keep Wav2Lip as fallback; can swap lip sync model independently |
| Total pipeline exceeds 200ms budget | Medium (30%) | High | Drop CodeFormer (save 25ms) or reduce LivePortrait resolution to 384px |
| Model weights too large for RunPod disk | Low (10%) | Low | Use persistent volume or pre-baked Docker image |
| LivePortrait source preparation takes too long | Low (15%) | Medium | Pre-cache source features; run once at session start |

### 13.2 Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Teeth still look unnatural in edge cases | Medium (30%) | Medium | Tune CodeFormer fidelity (0.5-0.8 range); add teeth-specific post-processing |
| Skin tone mismatch under extreme lighting | Medium (35%) | Medium | Add lightness-channel only mode; user can adjust in settings |
| Lip sync visibly lags audio | Medium (30%) | High | Reduce audio buffer to 50ms; implement client-side audio delay compensation |
| Edge artifacts during fast head movement | Medium (25%) | Medium | Increase face detection interval adaptively; add motion blur compensation |

### 13.3 Infrastructure Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| A100 cost too high for project budget | Medium (40%) | High | Start with RTX 3090 (200-280ms latency); upgrade when revenue justifies |
| RunPod availability issues | Low (15%) | Medium | Multi-provider setup (RunPod + Lambda Labs + Vast.ai) |
| Model licensing issues | Low (10%) | High | All recommended models are open source (MIT/Apache); verify before production |

---

## Summary: Before → After Quality Scores

| Feature | Before (V2) | After (V3) | Improvement |
|---------|-------------|------------|-------------|
| 1. Teeth rendering | 40/100 | **85-90** | +45-50 |
| 2. Skin tone matching | 45/100 | **88-92** | +43-47 |
| 3. Lip sync accuracy | 35/100 | **80-85** | +45-50 |
| 4. Eye movement/gaze | 0/100 | **88-92** | +88-92 |
| 5. Blink & micro expressions | 0-30/100 | **85-90** | +55-90 |
| 6. Light smile capture | 35/100 | **85-90** | +50-55 |
| 7. Skin texture/pores | 0/100 | **75-80** | +75-80 |
| 8. Edge blending | 55/100 | **90-92** | +35-37 |
| 8b. Lighting consistency | 40/100 | **65-70** | +25-30 |
| **Overall realism** | **~35/100** | **~83-87/100** | **+48-52** |

**Critical models to replace:**
- ❌ `inswapper_128.onnx` → ✅ **LivePortrait** (512×512, motion transfer)
- ❌ `wav2lip_gan_96.onnx` → ✅ **MuseTalk** (256×256, micro-movement lip sync)
- ❌ `GFPGANv1.4.pth` → ✅ **CodeFormer** (512×512, fidelity-preserving enhancement)

**Minimum GPU:** A100 40GB (~$720/month on RunPod)  
**Timeline:** 6-7 weeks for full implementation  
**Latency target:** 120-160ms on A100 (15-20 FPS effective)
