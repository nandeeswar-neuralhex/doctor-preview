---
applyTo: '**'
---
# 💬 Prompt Templates — Quick Triggers

Pre-built prompt patterns to activate the SDLC pipeline for common tasks.
Copy-paste or reference these when triggering agents.

---

## Template 1: New Feature (Full Pipeline)

```
REQUIREMENT: [Describe the feature]

Example:
  "Add a booking management system where patients can book consultation slots,
   doctors can view their schedule, and the system sends confirmation notifications."

This triggers: Agent 1 → 2 → 3 → 4 (full SDLC pipeline)
```

---

## Template 2: Bug Fix (Targeted)

```
BUG: [Describe the bug]
OBSERVED: [What actually happens]
EXPECTED: [What should happen]
REPRODUCE: [Steps to reproduce]

Example:
  BUG: Face swap flickers when user moves head quickly
  OBSERVED: The swapped face jitters and shows original face for 1-2 frames
  EXPECTED: Smooth face replacement regardless of head movement speed
  REPRODUCE: Start face swap → move head left-right rapidly → observe output

This triggers: Agent 1 (analysis only) → Agent 2 (fix) → Agent 3 (verify fix)
```

---

## Template 3: API Endpoint Only (Backend)

```
ADD API: [HTTP method] /api/v1/[resource]
PURPOSE: [What it does]
INPUT: { field: type, ... }
OUTPUT: { field: type, ... }
ERRORS: [Expected error cases]

Example:
  ADD API: POST /api/v1/sessions
  PURPOSE: Create a new face swap session
  INPUT: { "target_image": "base64string", "quality": "high" | "medium" | "low" }
  OUTPUT: { "session_id": "uuid", "status": "ready", "created_at": "iso8601" }
  ERRORS: 400 (invalid image), 413 (image too large), 500 (GPU init failed)

This triggers: Agent 2 (code) → Agent 3 (test)
```

---

## Template 4: UI Component Only (Frontend)

```
ADD UI: [Component name]
PURPOSE: [What user sees/does]
DATA SOURCE: [API endpoint or local state]
STATES: [Loading, Error, Empty, Populated]

Example:
  ADD UI: SessionDashboard
  PURPOSE: Show active face swap sessions with status, start/stop controls
  DATA SOURCE: GET /api/v1/sessions
  STATES: Loading spinner, "No active sessions" empty, session cards with status badges

This triggers: Agent 4 (UI only, assumes API exists)
```

---

## Template 5: Refactor (Code Quality)

```
REFACTOR: [What to refactor]
REASON: [Why — DRY violation, SOLID violation, performance]
SCOPE: [Files affected]
CONSTRAINT: [No behavior change / specific behavior change]

Example:
  REFACTOR: Extract face detection caching from face_swapper.py
  REASON: face_swapper.py is 1100+ lines, detection + caching is a separate concern (SRP)
  SCOPE: runpod_v2/src/face_swapper.py → new runpod_v2/src/face_detector.py
  CONSTRAINT: No behavior change — same API, same smoothing behavior

This triggers: Agent 1 (impact analysis) → Agent 2 (refactor) → Agent 3 (verify no regression)
```

---

## Template 6: Performance Optimization

```
OPTIMIZE: [What is slow]
CURRENT: [Current performance metric]
TARGET: [Target metric]
PROFILING: [Where bottleneck is, if known]

Example:
  OPTIMIZE: WebSocket frame processing latency
  CURRENT: 65ms per frame average
  TARGET: < 40ms per frame
  PROFILING: TurboJPEG decode takes 8ms, face detection 25ms, swap 20ms, encode 12ms

This triggers: Agent 1 (bottleneck analysis) → Agent 2 (optimization) → Agent 3 (benchmark)
```

---

## Template 7: Integration (Connect Existing Pieces)

```
INTEGRATE: [System A] with [System B]
PURPOSE: [Why they need to connect]
DATA FLOW: [A] → [transform] → [B]
PROTOCOL: [HTTP/WebSocket/WebRTC/Event]

Example:
  INTEGRATE: Clerk authentication with face swap sessions
  PURPOSE: Only authenticated users can create sessions
  DATA FLOW: Clerk JWT → validate on server → attach user_id to session
  PROTOCOL: HTTP header (Authorization: Bearer <token>)

This triggers: Agent 1 → Agent 2 → Agent 3 → Agent 4 (if UI changes needed)
```

---

## Template 8: Database Feature

```
ADD DATA: [Entity/Collection name]
PURPOSE: [What data is stored]
OPERATIONS: [CRUD operations needed]
STORAGE: [In-memory / SQLite / Cosmos DB]

Example:
  ADD DATA: consultation_history
  PURPOSE: Store completed face swap consultations for patient records
  OPERATIONS: Create (after session), Read (patient history), List (doctor dashboard)
  STORAGE: Azure Cosmos DB (partition key: patient_id)

This triggers: Full pipeline with data modeling focus
```

---

## Template 9: Deploy/DevOps

```
DEPLOY: [What to deploy]
TARGET: [RunPod / Azure VM / Static Web App]
CONFIG: [What needs to change]

Example:
  DEPLOY: Updated face_swapper.py to RunPod
  TARGET: RunPod GPU pod (A40 80GB)
  CONFIG: Update Dockerfile, rebuild image, restart pod

This triggers: Deployment checklist + verification
```

---

## Template 10: Test Only (Verify Existing)

```
TEST: [What to test]
ENDPOINTS: [List of endpoints]
SCENARIOS: [Specific test cases]

Example:
  TEST: All face swap API endpoints after lip sync refactor
  ENDPOINTS: POST /swap, POST /set-target, GET /health, DELETE /session/{id}
  SCENARIOS: Happy path, invalid image, no target set, concurrent sessions

This triggers: Agent 3 only (testing)
```

---

## Quick Commands

| Say This | Pipeline |
|---|---|
| "New feature: ..." | Full: 1 → 2 → 3 → 4 |
| "Fix bug: ..." | Targeted: 1 → 2 → 3 |
| "Add endpoint: ..." | Backend: 2 → 3 |
| "Add component: ..." | Frontend: 4 |
| "Refactor: ..." | Quality: 1 → 2 → 3 |
| "Optimize: ..." | Performance: 1 → 2 → 3 |
| "Test: ..." | Testing only: 3 |
| "Deploy: ..." | DevOps checklist |
