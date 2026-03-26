# 🎯 Real-Time Face Replacement — Complete Roadmap

## Vision
Transform live webcam video so the source person is completely replaced by the target person — face, hair, expressions, lip sync, and eventually full body — at 30fps with zero visible artifacts, indistinguishable from a real video call.

---

## Current State (March 2026)

### What We Have
| Component | Status | Technology |
|---|---|---|
| Face swap engine | ✅ Working | INSwapper 128×128 ONNX |
| Face detection (real-time) | ✅ Working | RetinaFace 320×320 (buffalo_l, detection-only) |
| Face detection (upload) | ✅ Working | buffalo_l 640×640 (full analysis) |
| Lip sync | ✅ Working | Wav2Lip 96×96 ONNX |
| Face enhancement | ✅ Loaded (not applied real-time) | GFPGAN v1.4 |
| Temporal smoothing | ✅ Working (15 fixes) | EMA on bbox, kps, affine, LUT, mouth |
| Mask/boundary | ⚠️ Oval ellipse | GaussianBlur on soft oval |
| Streaming (WebRTC) | ✅ Working | aiortc + VP8 patched 3Mbps |
| Streaming (WebSocket) | ✅ Working | Binary pipeline + jitter buffer |
| Desktop app | ✅ Working | Electron + React + canvas |
| Deployment | ✅ Working | RunPod GPU (T4/A10/H100) |

### Current Quality Score: **80-85 / 100**

### Known Limitations
1. Oval mask — does not follow hair/ear/jawline contour
2. 128×128 resolution — soft/blurry on close-up
3. Wav2Lip 96×96 — low-res mouth generation
4. No hair replacement — original person's hair visible
5. VP8 codec — compression artifacts at fast movement
6. 200-400ms latency on India→US connection

---

## PHASE 1: Enhanced Face Swap
**Timeline:** 2-3 weeks  
**Quality Target:** 90 / 100  
**Risk Level:** Low — all components exist, just integration  

### 1.1 BiSeNet Face Parsing Model
**Priority: 🔴 CRITICAL — Single biggest quality improvement**

**What:** Add a face parsing segmentation model that classifies every pixel as:
- Background, Skin, Nose, Eyes, Eyebrows, Ears, Mouth, Upper lip, Lower lip, Hair, Hat, Earring, Necklace, Neck, Cloth

**Why:** Replaces the oval ellipse mask with a pixel-perfect face boundary. Hair, ears, jawline, beard — all precisely separated from background.

**Technical Plan:**
```
Model:    BiSeNet (face parsing) — pretrained on CelebAMask-HQ
Format:   ONNX (export from PyTorch)
Input:    512×512 RGB image
Output:   512×512 segmentation map (19 classes)
Speed:    ~3-5ms on H100, ~8-12ms on T4
Memory:   ~50MB model file
```

**Implementation Steps:**
1. Download pretrained BiSeNet ONNX model (~50MB)
2. Create `face_parser.py` module:
   - `parse(frame, face_bbox) → mask` — returns binary mask for face+hair+ears+neck
   - Configurable classes: can include/exclude hair, neck, ears
   - Cache parsing result (changes slowly between frames)
3. Modify `_swap_single_face()` in `face_swapper.py`:
   - Replace oval mask generation with BiSeNet parsing mask
   - Apply temporal smoothing to parsing mask (prevents boundary flicker)
   - Use GaussianBlur on parsing boundary only (1-2px feather)
4. Add to `download_models.py` for auto-download
5. Add `ENABLE_FACE_PARSING` config flag (fallback to oval)

**Files to Create/Modify:**
- CREATE: `runpod_v2/src/face_parser.py`
- MODIFY: `runpod_v2/src/face_swapper.py` — use parsing mask
- MODIFY: `runpod_v2/src/config.py` — add ENABLE_FACE_PARSING
- MODIFY: `runpod_v2/src/download_models.py` — add BiSeNet model
- MODIFY: `runpod_v2/requirements.txt` — if any new deps
- MODIFY: `runpod_v2/Dockerfile` — include model in container

**Acceptance Criteria:**
- [ ] Face boundary follows actual face contour (not oval)
- [ ] Ears visible when head turns — boundary follows ear edge
- [ ] Beard boundary precise — no background bleeding into beard
- [ ] Hair boundary clean — no halo effect
- [ ] Works at 30fps on H100 (< 5ms overhead)
- [ ] Fallback to oval mask if parsing fails

---

### 1.2 Enable GFPGAN Real-Time Enhancement
**Priority: 🟡 HIGH — Already loaded, just needs activation**

**What:** Apply GFPGAN face enhancement to the swapped face to recover HD skin texture lost in 128×128 upscale.

**Why:** INSwapper outputs 128×128 which looks soft when upscaled to ~250px face region. GFPGAN adds realistic skin pores, eye detail, lip texture.

**Technical Plan:**
```
Model:    GFPGANv1.4 (already loaded in FaceSwapper.__init__)
Input:    Face crop (BGR)
Output:   Enhanced face crop (same size, sharper details)
Speed:    ~8-12ms on H100, ~25-35ms on T4
Trigger:  Apply AFTER INSwapper, BEFORE paste-back
```

**Implementation Steps:**
1. In `_swap_single_face()`, after `bgr_fake` is produced:
   - Upscale `bgr_fake` from 128×128 to 256×256 or 512×512
   - Run through `self.enhancer.enhance()`
   - Downscale result to ROI size
   - Continue with normal paste-back
2. Add `ENABLE_REALTIME_ENHANCE` config flag
3. On slower GPUs (T4), skip enhancement — only on H100
4. Optional: run enhancement every Nth frame and blend (e.g., every 2nd frame)

**Files to Modify:**
- MODIFY: `runpod_v2/src/face_swapper.py` — add enhancement step
- MODIFY: `runpod_v2/src/config.py` — add ENABLE_REALTIME_ENHANCE

**Acceptance Criteria:**
- [ ] Skin texture visibly sharper in close-up
- [ ] No added flickering from enhancement
- [ ] Frame rate stays ≥ 28fps on H100
- [ ] Disabled by default on non-H100 GPUs
- [ ] Enhancement doesn't alter skin color (only texture)

---

### 1.3 Upgrade Detection to 640×640 on H100
**Priority: 🟡 HIGH — Better landmarks = smoother swap**

**What:** Use the full 640×640 face analyzer (already loaded as `face_analyzer`) for real-time detection when running on H100.

**Why:** 320×320 detection with 5-point landmarks gives noisy keypoints on small/distant faces. 640×640 with full analysis gives:
- More accurate 5-point keypoints → better affine warp
- Optional 68-point landmarks → could be used for mask contour
- Optional 106-point landmarks → fine expression features
- Better detection of partially occluded faces

**Implementation Steps:**
1. Add GPU capability detection at startup:
   - If H100/A100 (> 40GB VRAM), use 640×640 full analysis
   - If T4/A10 (< 24GB VRAM), keep 320×320 detection-only
2. Modify `_detect_faces_with_fallback()`:
   - Use `face_analyzer` (640×640) instead of `face_analyzer_fast`
   - Still apply det_score >= 0.4 filter
3. Optional: Extract 68-point landmarks and use for expression matching

**Files to Modify:**
- MODIFY: `runpod_v2/src/face_swapper.py` — conditional analyzer selection
- MODIFY: `runpod_v2/src/config.py` — add DETECTION_SIZE config

**Acceptance Criteria:**
- [ ] Smoother face tracking on head rotation
- [ ] Better detection of faces at distance
- [ ] No FPS regression on H100 (still ≥ 30fps)
- [ ] Graceful fallback to 320×320 on T4

---

### 1.4 VP8 → H.264 Codec Upgrade (WebRTC)
**Priority: 🟢 MEDIUM — Quality improvement at same bitrate**

**What:** Replace VP8 with H.264 for WebRTC encoding. H.264 is ~40% more efficient, meaning same quality at lower bitrate or better quality at same bitrate.

**Technical Plan:**
- aiortc supports H.264 via `openh264`
- Need to install `openh264` system library in Docker
- Modify VP8 encoder monkey-patch to H.264 equivalent
- Client negotiation should prefer H.264 automatically

**Implementation Steps:**
1. Add `openh264` to Dockerfile
2. Configure aiortc to prefer H.264 codec
3. Set encoding parameters: profile=baseline, level=3.1, bitrate=3Mbps
4. Test with desktop app client

**Files to Modify:**
- MODIFY: `runpod_v2/Dockerfile` — install openh264
- MODIFY: `runpod_v2/src/webrtc.py` — H.264 encoder config
- MODIFY: `runpod_v2/requirements.txt` — if needed

**Acceptance Criteria:**
- [ ] WebRTC uses H.264 when available
- [ ] Falls back to VP8 if client doesn't support H.264
- [ ] Visibly better quality at 3Mbps (less blocking)
- [ ] Same or better latency

---

### Phase 1 Summary

| Before Phase 1 | After Phase 1 |
|---|---|
| Oval ellipse mask | Pixel-perfect BiSeNet mask |
| 128×128 soft face | HD face texture via GFPGAN |
| 320×320 noisy landmarks | 640×640 precise landmarks |
| VP8 compression artifacts | H.264 clean video |
| **Quality: 80-85/100** | **Quality: 90/100** |

**Total Additional Latency on H100: ~15-20ms**  
**Target FPS: 28-33fps ✅**

---

## PHASE 2: Full Head Replacement (OPTIONAL — Can Skip to Phase 3)
**Timeline:** 3-4 weeks  
**Quality Target:** 94 / 100  
**Risk Level:** Medium — requires multi-angle target capture  
**Recommendation:** Skip this and go straight to Phase 3 (LivePortrait)

### What Phase 2 Would Do
- Warp target person's FULL HEAD (including hair, beard, neck) onto source
- Requires 20-30 photos of target from multiple angles
- More complex warping (thin-plate spline or 3D morphable model)
- Hair from target photo warped to match source head angle

### Why Skip Phase 2
Phase 3 (LivePortrait) achieves better results with:
- Only 1 photo needed (vs 20-30)
- Hair is GENERATED (not warped from photo) → natural flow
- Expressions are GENERATED → perfect match
- No complex multi-angle warping logic
- Less code to maintain

**Phase 2 is documented here for completeness but not recommended for implementation.**

---

## PHASE 3: Motion-Driven Face Generation (LivePortrait)
**Timeline:** 4-6 weeks  
**Quality Target:** 97 / 100  
**Risk Level:** Medium-High — new model architecture, but well-documented open source  

### 3.1 LivePortrait Integration

**What Is LivePortrait:**
LivePortrait (by Kuaishou) is a motion-driven portrait animation model that takes:
- **One target photo** (the person you want to appear as)
- **Driving motion** (extracted from webcam video frame-by-frame)

And **generates** a new image of the target person making the exact same expression, head pose, and eye gaze as the source person. It doesn't warp pixels — it generates new ones from scratch using a learned representation.

**Why LivePortrait:**
```
✅ Open source (Apache 2.0 license)
✅ ONNX export supported
✅ ~30ms inference on H100 → 30fps
✅ Handles extreme expressions (laughing, shouting, winking)
✅ Hair animates naturally with head movement
✅ Eye gaze direction preserved
✅ Only 1 target photo needed
✅ No Wav2Lip needed — lips are correct by generation
✅ Production-tested by Kuaishou (billions of users)
```

**Architecture:**
```
┌──────────────┐    ┌──────────────┐
│  Source       │    │  Target      │
│  Webcam Frame │    │  Photo       │
└──────┬───────┘    └──────┬───────┘
       │                    │
       ▼                    ▼
┌──────────────┐    ┌──────────────┐
│  Motion      │    │  Appearance  │
│  Extractor   │    │  Extractor   │
│  (keypoints, │    │  (identity,  │
│   expression,│    │   texture,   │
│   head pose) │    │   features)  │
└──────┬───────┘    └──────┬───────┘
       │                    │
       ▼                    ▼
       ┌────────────────────┐
       │   Motion Warping   │
       │   + Generation     │
       │   Network          │
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  Generated Frame   │
       │  (target person    │
       │   with source      │
       │   motion/expression│
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  BiSeNet Parsing   │
       │  Mask + Paste      │
       │  onto source body  │
       │  + background      │
       └────────────────────┘
```

### 3.2 Implementation Plan

**Step 1: Model Setup (Week 1)**
```
Tasks:
  1. Clone LivePortrait repo, export models to ONNX
     - Appearance encoder → ONNX (~100MB)
     - Motion extractor → ONNX (~50MB)
     - Warping + generator → ONNX (~200MB)
     - Stitching module → ONNX (~20MB)
  2. Add models to download_models.py
  3. Create live_portrait.py module:
     - __init__: load all ONNX sessions on GPU
     - extract_appearance(target_photo) → appearance_features
     - extract_motion(source_frame) → motion_keypoints
     - generate(appearance, motion) → generated_face
  4. Benchmark on H100: target < 30ms total
```

**Step 2: Pipeline Integration (Week 2)**
```
Tasks:
  1. Modify server.py and webrtc.py:
     - On target photo upload: extract appearance features (one-time)
     - On each webcam frame: extract motion → generate → paste
  2. Replace INSwapper call with LivePortrait generate:
     Current:  swapper.get(frame, source, target, paste_back=False) → 128x128
     New:      live_portrait.generate(appearance, motion) → 256x256 or 512x512
  3. Use BiSeNet parsing mask (Phase 1) for paste-back
  4. Remove Wav2Lip dependency (lips are correct by generation)
  5. Keep temporal smoothing on motion keypoints (EMA)
```

**Step 3: Quality Tuning (Week 3)**
```
Tasks:
  1. Motion keypoint smoothing:
     - Apply SMOOTHING_ALPHA EMA to motion keypoints
     - Prevents jitter in generated face
  2. Appearance blending:
     - If multiple target photos uploaded, extract all appearances
     - Blend appearance features based on head angle match
     - Front photo for 0-15°, side photos for 15-45°
  3. Stitching optimization:
     - LivePortrait has a "stitching module" that blends generated face
       with background seamlessly
     - Tune stitching parameters for invisible boundary
  4. Eye gaze correction:
     - LivePortrait can adjust eye gaze
     - Configure to look at camera (important for video calls)
```

**Step 4: Performance Optimization (Week 4)**
```
Tasks:
  1. CUDA graph capture for motion extractor (skip kernel launch overhead)
  2. TensorRT optimization of generator (potential 2x speedup)
  3. Batch appearance extraction on upload (precompute all angles)
  4. Pipeline GPU operations: overlap motion extraction with generation
  5. Target: < 25ms per frame on H100 → 40fps headroom
```

### 3.3 Files to Create/Modify

| Action | File | Description |
|---|---|---|
| CREATE | `runpod_v2/src/live_portrait.py` | Main LivePortrait wrapper (~300 lines) |
| CREATE | `runpod_v2/src/motion_extractor.py` | Motion keypoint extraction (~150 lines) |
| MODIFY | `runpod_v2/src/server.py` | Use LivePortrait instead of INSwapper |
| MODIFY | `runpod_v2/src/webrtc.py` | Use LivePortrait in video transform |
| MODIFY | `runpod_v2/src/config.py` | Add SWAP_ENGINE = "liveportrait" / "inswapper" |
| MODIFY | `runpod_v2/src/download_models.py` | Add LivePortrait model downloads |
| MODIFY | `runpod_v2/Dockerfile` | Include LivePortrait model files |
| MODIFY | `runpod_v2/requirements.txt` | Add dependencies |
| REMOVE | Wav2Lip dependency | Lips are correct by generation |

### 3.4 Input Requirements

**Minimum:** 1 high-quality front-facing photo (512×512 or larger)

**Recommended (5 photos):**
| Photo | Purpose |
|---|---|
| Front face, neutral expression | Primary appearance source |
| Front face, smiling | Expression range reference |
| Slight left turn (~20°) | Side appearance for head rotation |
| Slight right turn (~20°) | Side appearance for head rotation |
| Different lighting | Lighting adaptation |

**Photo Quality Guidelines:**
- Resolution: At least 512×512 face region
- Lighting: Even, front-lit preferred
- Expression: Neutral for primary, varied for extras
- Background: Any (will be cropped to face region)
- Format: JPEG or PNG

### 3.5 What This Phase Removes

| Removed Component | Replaced By |
|---|---|
| INSwapper 128×128 | LivePortrait 256×256 generator |
| Wav2Lip 96×96 | LivePortrait generates correct lips |
| Oval mask blending | BiSeNet parsing mask (Phase 1) |
| Expression matching (multi-target) | Motion-driven generation (automatic) |
| Color LUT matching | Generator produces correct skin tone |
| Mouth preservation logic | Not needed — mouth is generated correctly |

### Phase 3 Summary

| Before Phase 3 | After Phase 3 |
|---|---|
| Warp 128×128 photo pixels | Generate 256×256 new pixels per frame |
| Wav2Lip lip sync (low res) | Lips correct by generation |
| Oval/parsing mask boundary | Generated face + stitching module |
| Need 10+ expression photos | Need only 1 photo |
| Hair from original person | Target person's hair, animated |
| **Quality: 90/100** | **Quality: 97/100** |

---

## PHASE 4: Full Body Replacement (Future)
**Timeline:** 6-8 weeks (after Phase 3 is stable)  
**Quality Target:** 99 / 100  
**Risk Level:** High — cutting-edge technology, some components still maturing  

### 4.1 Overview

Full body replacement means: the source person's ENTIRE visible body is replaced with the target person, including:
- Face and hair (Phase 3)
- Clothing and accessories
- Body shape and proportions
- Hand gestures and arm movements
- Sitting/standing posture

### 4.2 Technology Stack

```
Source webcam frame
  │
  ├─► DWPose / MMPose: Extract body pose (33+ keypoints)
  │     - Shoulders, elbows, wrists, fingers
  │     - Torso, hips
  │     - Head position and angle
  │
  ├─► LivePortrait: Face generation (Phase 3)
  │
  ├─► Body Generation Model (one of):
  │     Option A: AnimateAnyone (Alibaba) — image-to-video conditioned on pose
  │     Option B: MagicAnimate (ByteDance) — similar approach
  │     Option C: ControlNet + SDXL Turbo — diffusion-based, real-time
  │     Option D: MuseV (Tencent) — multi-reference generation
  │
  └─► Compositing:
        - Generated body + generated face
        - Background from original webcam frame
        - Boundary blending at body silhouette
```

### 4.3 Performance Estimate on H100

| Component | Latency | Notes |
|---|---|---|
| Body pose extraction (DWPose) | ~5ms | Lightweight model |
| Face generation (LivePortrait) | ~25ms | Phase 3 |
| Body generation | ~50-80ms | Main bottleneck |
| Compositing | ~3ms | GPU-accelerated |
| **Total** | **~83-113ms** | **~9-12fps** |

### 4.4 Feasibility Assessment

| Aspect | Status | Notes |
|---|---|---|
| Pose extraction | ✅ Production-ready | DWPose/MMPose are stable |
| Face generation | ✅ Production-ready | LivePortrait (Phase 3) |
| Body generation | ⚠️ Maturing | AnimateAnyone closest to production |
| Hand detail | ⚠️ Challenging | Fingers often distorted |
| Clothing realism | ⚠️ Good but imperfect | Wrinkles/folds not always accurate |
| Real-time (30fps) | ❌ Not yet | ~9-12fps current best on H100 |
| Multi-GPU pipeline | ✅ Possible | Face on GPU0, body on GPU1 → 20fps |

### 4.5 Practical Approach for Video Calls

**For video calls specifically,** full body replacement may not be necessary because:
- Most video calls show **head and upper torso only**
- Clothing below the neck can be hidden with a **virtual background**
- Phase 3 (head + hair) covers 90% of visible area in a typical call

**Recommended hybrid approach:**
```
Phase 3 (head replacement) + Virtual clothing overlay at collar/shoulders
  = Visually complete replacement for video call scenarios
  = 30fps achievable on single H100
```

### 4.6 When to Implement Phase 4
- After Phase 3 is deployed and stable (2+ months)
- When body generation models reach 30fps on H100 (likely late 2026)
- When specific use cases require full body visibility
- Consider multi-GPU deployment (2× H100) for 20fps body generation

---

## Implementation Timeline

```
Week 1-2:   Phase 1.1 — BiSeNet face parsing integration
Week 2-3:   Phase 1.2 — GFPGAN real-time enhancement
Week 3:     Phase 1.3 — 640×640 detection on H100
Week 3:     Phase 1.4 — H.264 codec upgrade
            ─────── DEPLOY Phase 1 → Quality: 90/100 ───────
Week 4-5:   Phase 3.1 — LivePortrait model setup + ONNX export
Week 5-6:   Phase 3.2 — Pipeline integration (replace INSwapper)
Week 7:     Phase 3.3 — Quality tuning + eye gaze
Week 8:     Phase 3.4 — Performance optimization (TensorRT)
            ─────── DEPLOY Phase 3 → Quality: 97/100 ───────
Week 12+:   Phase 4 — Full body (when technology matures)
            ─────── DEPLOY Phase 4 → Quality: 99/100 ───────
```

---

## Hardware Requirements

| Phase | GPU | VRAM | CPU | RAM |
|---|---|---|---|---|
| Phase 1 | H100 (recommended) or A10 | 16GB+ | 4+ vCPU | 16GB+ |
| Phase 3 | H100 (required for 30fps) | 40GB+ | 8+ vCPU | 32GB+ |
| Phase 4 | 2× H100 (for 20fps) | 80GB+ each | 16+ vCPU | 64GB+ |

---

## Model Files Summary

| Model | Size | Phase | Purpose |
|---|---|---|---|
| inswapper_128.onnx | ~500MB | Current | Face swap (128×128) |
| buffalo_l (RetinaFace) | ~250MB | Current | Face detection |
| wav2lip_gan_96.onnx | ~100MB | Current | Lip sync (removed in Phase 3) |
| GFPGANv1.4.pth | ~330MB | Phase 1 | Face enhancement |
| bisenet_face_parsing.onnx | ~50MB | Phase 1 | Face segmentation |
| liveportrait_appearance.onnx | ~100MB | Phase 3 | Appearance encoder |
| liveportrait_motion.onnx | ~50MB | Phase 3 | Motion extractor |
| liveportrait_generator.onnx | ~200MB | Phase 3 | Image generator |
| liveportrait_stitching.onnx | ~20MB | Phase 3 | Seamless blending |
| dwpose.onnx | ~100MB | Phase 4 | Body pose extraction |
| body_generator.onnx | ~500MB+ | Phase 4 | Full body generation |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| LivePortrait ONNX export fails | Low | High | Use PyTorch inference as fallback |
| H100 not fast enough for Phase 3 | Low | High | TensorRT optimization, reduce resolution |
| LivePortrait quality worse than INSwapper | Low | High | Keep INSwapper as fallback engine (config switch) |
| BiSeNet parsing too slow on T4 | Medium | Medium | Run every 3rd frame, interpolate mask |
| GFPGAN changes skin color | Medium | Low | Add color correction pass after enhancement |
| Phase 4 body generation artifacts | High | Medium | Limit to head+shoulders replacement for now |
| Model licensing issues | Low | High | All recommended models are Apache 2.0 / MIT |

---

## Configuration Design

```python
# config.py additions for multi-phase support

# Swap Engine Selection
SWAP_ENGINE = os.getenv("SWAP_ENGINE", "inswapper")  # "inswapper" | "liveportrait"

# Phase 1: Face Parsing
ENABLE_FACE_PARSING = os.getenv("ENABLE_FACE_PARSING", "true").lower() == "true"
FACE_PARSING_MODEL = os.path.join(MODELS_DIR, "bisenet_face_parsing.onnx")
PARSING_CLASSES = ["face", "hair", "ears", "neck"]  # Which regions to include in mask

# Phase 1: Real-time Enhancement
ENABLE_REALTIME_ENHANCE = os.getenv("ENABLE_REALTIME_ENHANCE", "false").lower() == "true"
ENHANCE_EVERY_N_FRAMES = int(os.getenv("ENHANCE_EVERY_N_FRAMES", "1"))

# Phase 1: Detection Resolution
DETECTION_SIZE = int(os.getenv("DETECTION_SIZE", "320"))  # 320 for T4, 640 for H100

# Phase 3: LivePortrait
LIVEPORTRAIT_MODEL_DIR = os.path.join(MODELS_DIR, "liveportrait")
LIVEPORTRAIT_RESOLUTION = int(os.getenv("LIVEPORTRAIT_RESOLUTION", "256"))  # 256 or 512
ENABLE_EYE_GAZE_CORRECTION = os.getenv("ENABLE_EYE_GAZE_CORRECTION", "true").lower() == "true"
MOTION_SMOOTHING_ALPHA = float(os.getenv("MOTION_SMOOTHING_ALPHA", "0.7"))
```

---

## Success Metrics

| Metric | Phase 1 Target | Phase 3 Target | Phase 4 Target |
|---|---|---|---|
| Swap FPS | ≥ 28 fps | ≥ 28 fps | ≥ 20 fps |
| Face boundary quality | 9/10 (parsing) | 10/10 (generated) | 10/10 |
| Expression accuracy | 7/10 (warp) | 9.5/10 (generated) | 9.5/10 |
| Lip sync accuracy | 7/10 (Wav2Lip) | 9/10 (generated) | 9/10 |
| Hair realism | 5/10 (original) | 9/10 (generated) | 9/10 |
| Skin texture | 8/10 (GFPGAN) | 9/10 (generated) | 9/10 |
| Body replacement | N/A | N/A | 8/10 |
| Overall realism | **90/100** | **97/100** | **99/100** |

---

## Quick Start — What to Build First

**Start with Phase 1.1 (BiSeNet face parsing)** — it's the single biggest visual improvement and all other phases build on it. The parsing mask is used in Phase 1, Phase 3, and Phase 4.

```
Next Action: Implement BiSeNet face parsing in face_swapper.py
Expected Time: 3-4 days
Expected Impact: Face boundary quality jumps from 5/10 to 9/10
```
