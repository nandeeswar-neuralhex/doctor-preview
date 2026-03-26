---
applyTo: '**'
---
# 🎯 SDLC Orchestrator — Master Agent

## Overview
This is the master orchestration agent that manages the full Software Development Lifecycle. When a user provides a requirement, this agent coordinates 4 specialized agents in sequence, with automatic retry loops on failure.

## Agent Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    USER REQUIREMENT                             │
│  "Add feature X", "Build Y", "Implement Z"                    │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  AGENT 1: REQUIREMENT ANALYSIS                                 │
│  ─────────────────────────────                                 │
│  • Parse requirement into user stories                         │
│  • Define API contracts (endpoints, request/response)          │
│  • Identify all files to create/modify                         │
│  • Define acceptance criteria                                  │
│  • Risk assessment                                             │
│                                                                │
│  Output: Analysis Document                                     │
│  Gate: User confirms "Proceed to coding"                       │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  AGENT 2: CODE GENERATION                                      │
│  ────────────────────────                                      │
│  • Backend: Models → Services → API endpoints                  │
│  • Frontend: Hooks → Components → Integration                  │
│  • Follow SOLID, DRY, logging, error handling                  │
│  • Type hints, docstrings, validation                          │
│                                                                │
│  Output: Production-ready code in workspace                    │
│  Gate: Auto-proceed to testing                                 │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  AGENT 3: API TESTING                                          │
│  ───────────────────────                                       │
│  • Generate curl commands for every endpoint                   │
│  • Run happy path + error path + edge case tests               │
│  • Execute in terminal, capture results                        │
│  • If ANY test fails:                                          │
│      → Diagnose failure                                        │
│      → Fix code (invoke Agent 2 for specific fix)              │
│      → Re-run failed tests                                     │
│      → Repeat until ALL pass (max 5 retries)                   │
│                                                                │
│  Output: Test report (all green)                               │
│  Gate: All tests passing → proceed to UI                       │
│                                                                │
│  ┌──────────────────────────────────────────────────┐          │
│  │           RETRY LOOP (max 5 attempts)            │          │
│  │                                                  │          │
│  │   Test Failed ──► Diagnose ──► Fix Code          │          │
│  │        ▲                          │               │          │
│  │        │                          ▼               │          │
│  │        └──────── Re-test ◄────────┘               │          │
│  │                                                  │          │
│  │   All Pass ──► Exit loop ──► Proceed             │          │
│  └──────────────────────────────────────────────────┘          │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  AGENT 4: UI INTEGRATION                                       │
│  ──────────────────────────                                    │
│  • Build UI components (React + Tailwind)                      │
│  • Create custom hooks for data management                     │
│  • Integrate with tested API endpoints                         │
│  • Add to existing app navigation/layout                       │
│  • End-to-end verification                                     │
│  • If E2E fails:                                               │
│      → Diagnose (is it UI bug? API bug? Integration bug?)      │
│      → Fix and re-verify                                       │
│                                                                │
│  Output: Complete working feature                              │
│  Gate: E2E verification passed                                 │
└───────────────────────┬────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────────────┐
│  ✅ FEATURE COMPLETE                                           │
│  ──────────────────────                                        │
│  • All code follows SOLID/DRY principles                       │
│  • All APIs tested and passing                                 │
│  • UI integrated and working                                   │
│  • Ready for commit                                            │
└────────────────────────────────────────────────────────────────┘
```

## How to Trigger the Pipeline

When a user provides a requirement, follow this exact sequence:

### Step 1: Announce Pipeline Start
```
═══════════════════════════════════════════
  🚀 SDLC PIPELINE ACTIVATED
  Feature: [extracted feature name]
  
  Phase 1/4: Requirement Analysis ... ⏳
═══════════════════════════════════════════
```

### Step 2: Run Agent 1 (Analysis)
Follow `.prompts/1-requirement-analysis.md` instructions completely.
Present analysis and wait for user confirmation.

### Step 3: Run Agent 2 (Code Generation)
```
  Phase 1/4: Requirement Analysis ... ✅
  Phase 2/4: Code Generation ......... ⏳
```
Follow `.prompts/2-code-generation.md` instructions completely.

### Step 4: Run Agent 3 (API Testing)
```
  Phase 2/4: Code Generation ......... ✅
  Phase 3/4: API Testing ............. ⏳
```
Follow `.prompts/3-api-testing.md` instructions completely.
If tests fail, enter retry loop (fix → retest → repeat).

### Step 5: Run Agent 4 (UI Integration)
```
  Phase 3/4: API Testing ............. ✅ (X/X tests, Y retries)
  Phase 4/4: UI Integration .......... ⏳
```
Follow `.prompts/4-ui-integration.md` instructions completely.

### Step 6: Complete
```
═══════════════════════════════════════════
  ✅ SDLC PIPELINE COMPLETE
  Feature: [feature name]
  
  Phase 1/4: Requirement Analysis ... ✅
  Phase 2/4: Code Generation ......... ✅ (X files created, Y modified)
  Phase 3/4: API Testing ............. ✅ (X/X tests, Y retries)
  Phase 4/4: UI Integration .......... ✅ (E2E verified)
  
  READY TO COMMIT
═══════════════════════════════════════════
```

## Quality Gates — Non-Negotiable

| Gate | Criteria | Blocks |
|---|---|---|
| Analysis → Code | User says "proceed" | Cannot code without approved analysis |
| Code → Test | All files created/modified | Cannot test incomplete code |
| Test → UI | ALL API tests passing | Cannot integrate with broken APIs |
| UI → Complete | E2E verification passed | Cannot mark complete without working UI |

## Retry Policy

| Agent | Max Retries | On Max Failure |
|---|---|---|
| Agent 2 (Code) | 3 | Ask user for clarification |
| Agent 3 (API Test) | 5 per endpoint | Report as BLOCKER, ask user |
| Agent 4 (UI) | 3 | Report as BLOCKER, ask user |

## Code Quality Checklist (Applied by Agent 2, Verified by Agent 3 & 4)

### Backend ✓
- [ ] Every module has `import logging; logger = logging.getLogger(__name__)`
- [ ] Every function has type hints and docstring
- [ ] Every API endpoint has Pydantic request/response models
- [ ] Every external call has try/except with specific exceptions
- [ ] No magic numbers — all constants named
- [ ] No code duplication — shared logic extracted
- [ ] Configuration via environment variables
- [ ] Consistent error response format: `{"error": "CODE", "message": "..."}`

### Frontend ✓
- [ ] Data fetching in custom hooks (not in components)
- [ ] Loading, error, and empty states handled
- [ ] Form validation on client side
- [ ] User feedback for all async operations
- [ ] No hardcoded URLs — use env vars
- [ ] Tailwind CSS (no inline styles)
- [ ] Accessible (proper labels, ARIA attributes)
- [ ] Responsive design

### API ✓
- [ ] All endpoints return correct HTTP status codes
- [ ] Validation errors return 422 with field-level details
- [ ] Not found returns 404
- [ ] Server errors return 500 with generic message (no stack traces)
- [ ] All endpoints tested with curl (happy + error paths)
- [ ] Response time < 500ms for non-GPU endpoints

## File Naming Conventions

```
Backend (Python):
  models/          — Data models, Pydantic schemas
  services/        — Business logic (one class per feature)
  routes/          — API endpoint definitions
  utils/           — Shared utilities
  config.py        — Configuration constants

Frontend (React):
  components/      — React components (PascalCase.jsx)
  hooks/           — Custom hooks (useCamelCase.js)
  utils/           — Helper functions
  constants/       — Shared constants

Naming:
  Python:  snake_case for files, functions, variables
           PascalCase for classes
  React:   PascalCase for components
           camelCase for hooks, variables, functions
  API:     kebab-case for URLs (/api/v1/my-resource)
  Config:  UPPER_SNAKE_CASE for constants
```
