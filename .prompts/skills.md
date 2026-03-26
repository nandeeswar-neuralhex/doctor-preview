---
applyTo: '**'
---
# 🛠️ Agent Skills — Reusable Capabilities

Skills are atomic, reusable capabilities any agent can invoke during the SDLC pipeline. Each skill has a clear trigger, input, process, and output.

---

## Skill Registry

| ID | Skill Name | Used By Agents | Description |
|---|---|---|---|
| SK-01 | Workspace Scanner | 1, 2 | Scan codebase to understand structure |
| SK-02 | Dependency Resolver | 2, 4 | Identify and install required packages |
| SK-03 | API Contract Generator | 1, 2 | Generate OpenAPI/Pydantic models from requirements |
| SK-04 | Test Runner | 3 | Execute curl tests and capture results |
| SK-05 | Retry Loop | 3, 4 | Auto-retry with diagnosis on failure |
| SK-06 | Code Validator | 2, 3 | Lint, type-check, syntax-check code |
| SK-07 | Git Committer | 0 (Orchestrator) | Stage, commit, push changes |
| SK-08 | Log Analyzer | 3, 4 | Parse server logs for errors |
| SK-09 | Performance Profiler | 2, 3 | Measure endpoint response times |
| SK-10 | File Impact Analyzer | 1, 2 | Find all files affected by a change |

---

## SK-01: Workspace Scanner

**Trigger**: Agent needs to understand project structure before making changes.

**Process**:
1. List the workspace root directory
2. Identify key directories: `src/`, `components/`, `hooks/`, `models/`, `services/`
3. Read `package.json` for frontend dependencies
4. Read `requirements.txt` for backend dependencies
5. Read `config.py` for project constants
6. Identify existing patterns (naming, structure, imports)

**Output**:
```
WORKSPACE SCAN:
  Backend:     runpod_v2/src/ (FastAPI, ONNX, InsightFace)
  Frontend:    desktop_app/src/ (React 18, Vite, Tailwind)
  Config:      runpod_v2/src/config.py
  Deployment:  azure_deployment/, runpod_service/
  Entry:       server.py (backend), App.jsx (frontend)
  Patterns:    snake_case (py), PascalCase (jsx), custom hooks (use*.js)
```

---

## SK-02: Dependency Resolver

**Trigger**: New feature requires packages not yet installed.

**Process**:
1. Check what new imports the feature needs
2. For Python: search in `requirements.txt`, install if missing
3. For JS: search in `package.json`, install if missing
4. Verify import works after installation

**Commands**:
```bash
# Python
cd runpod_v2 && pip install <package> && echo "<package>>=<version>" >> requirements.txt

# JavaScript
cd desktop_app && npm install <package>
```

**Output**: Updated requirements file + confirmation of successful import.

---

## SK-03: API Contract Generator

**Trigger**: New feature needs API endpoints defined.

**Process**:
1. From the requirement, identify resources and actions (CRUD)
2. Generate Pydantic request/response models
3. Define endpoint signatures with proper HTTP methods
4. Document error responses

**Template**:
```python
# models/<feature>.py
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class Create<Feature>Request(BaseModel):
    """Request body for creating a <feature>."""
    name: str = Field(..., min_length=1, max_length=255)
    # Add fields based on requirement

class <Feature>Response(BaseModel):
    """Response body for <feature> operations."""
    id: str
    name: str
    created_at: datetime
```

---

## SK-04: Test Runner

**Trigger**: Agent 3 needs to execute API tests.

**Process**:
1. Ensure server is running (check with health endpoint)
2. Execute each curl command in terminal
3. Capture HTTP status code and response body
4. Compare against expected values
5. Report pass/fail per test

**Health Check First**:
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/health
# Must return 200 before running tests
```

**Execution Pattern**:
```bash
# Run test and capture both body and status
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8765/api/v1/resource \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}')
BODY=$(echo "$RESPONSE" | head -n -1)
CODE=$(echo "$RESPONSE" | tail -1)
echo "Status: $CODE"
echo "Body: $BODY"
```

---

## SK-05: Retry Loop

**Trigger**: A test or verification step fails.

**Process**:
```
retry_count = 0
MAX_RETRIES = 5

while test_fails AND retry_count < MAX_RETRIES:
    1. Capture the error output
    2. Diagnose root cause:
       - Syntax error? → Fix the exact line
       - Logic error? → Fix the service/handler
       - Missing import? → Add the import
       - Wrong status code? → Fix the route/validation
       - Type error? → Fix the model/schema
    3. Apply the fix (invoke Code Generation for that specific file)
    4. Re-run the failing test
    5. retry_count += 1

if retry_count >= MAX_RETRIES:
    Report BLOCKER to user with:
    - What failed
    - All attempted fixes
    - Suggested manual investigation
```

**Output**: Either "Test now passing ✅" or "BLOCKER after 5 retries ❌".

---

## SK-06: Code Validator

**Trigger**: After generating or modifying code.

**Process**:
1. **Python syntax**: `python -m py_compile <file>`
2. **Python lint**: `python -m flake8 <file> --max-line-length=120`
3. **Python types**: `python -m mypy <file> --ignore-missing-imports`
4. **JS syntax**: Check for errors in VS Code diagnostics
5. **Import check**: Verify all imports resolve

**Quick Validate**:
```bash
# Python: syntax check
python -c "import ast; ast.parse(open('<file>').read()); print('OK')"

# Python: import check
python -c "from <module> import <name>; print('Import OK')"
```

---

## SK-07: Git Committer

**Trigger**: Feature complete, all tests passing.

**Process**:
1. Show `git diff --stat` for review
2. Stage relevant files: `git add <files>`
3. Commit with conventional format: `git commit -m "type: description"`
4. Push to current branch: `git push origin nandeeswar-webrtc-uma-fix`

**Commit Types**:
| Prefix | When |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructure (no behavior change) |
| `docs:` | Documentation only |
| `test:` | Test additions/changes |
| `chore:` | Build/tooling changes |

---

## SK-08: Log Analyzer

**Trigger**: Test fails and error isn't obvious from response.

**Process**:
1. Check server terminal output for Python tracebacks
2. Search for `ERROR`, `WARNING`, `Exception` in output
3. Parse the traceback to find:
   - File and line number of error
   - Exception type and message
   - Call stack leading to error
4. Report findings to the diagnosing agent

**Pattern**:
```bash
# Tail recent server logs
# Look for errors after the failed request timestamp
```

---

## SK-09: Performance Profiler

**Trigger**: Need to verify endpoint meets latency targets.

**Process**:
```bash
# Measure endpoint response time
curl -s -o /dev/null -w "Time: %{time_total}s\n" \
  -X GET http://localhost:8765/api/v1/resource

# Run 10 requests and average
for i in {1..10}; do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8765/api/v1/resource
done | awk '{sum+=$1} END {printf "Avg: %.3fs\n", sum/NR}'
```

**Targets**:
| Endpoint Type | Max Latency |
|---|---|
| Health check | 50ms |
| CRUD operation | 200ms |
| Face swap frame | 50ms (GPU) |
| WebRTC frame | 33ms (30fps) |

---

## SK-10: File Impact Analyzer

**Trigger**: Before implementing a feature, find all affected files.

**Process**:
1. Search workspace for related code:
   - Grep for function/class names that will change
   - Grep for import paths that will be affected
   - Grep for API endpoint URLs being modified
2. List all files grouped by change type:

**Output**:
```
FILES TO CREATE:
  - path/to/new_file.py — [purpose]

FILES TO MODIFY:
  - path/to/existing.py — [what changes]
  - path/to/component.jsx — [what changes]

FILES UNAFFECTED (verified):
  - path/to/safe_file.py — no references to changed code

RISK AREAS:
  - path/to/critical.py — imports from changed module, verify after change
```
