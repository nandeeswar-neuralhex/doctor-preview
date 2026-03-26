# GitHub Copilot Instructions — Doctor Preview

## Project Context
This is a **real-time face swap application** for medical consultation previews.
- Desktop client: Electron + React 18 + Tailwind CSS
- GPU server: FastAPI + InsightFace + ONNX + aiortc (WebRTC)
- Deployment: RunPod GPU / Azure VM

## Agent System
This project uses an SDLC agent orchestration system defined in `.prompts/`:

| File | Purpose |
|---|---|
| `.prompts/0-sdlc-orchestrator.md` | Master pipeline — coordinates all agents |
| `.prompts/1-requirement-analysis.md` | Parse requirements into technical stories |
| `.prompts/2-code-generation.md` | Generate code following SOLID/DRY/logging |
| `.prompts/3-api-testing.md` | Test APIs with curl, auto-retry on failure |
| `.prompts/4-ui-integration.md` | Build React UI, integrate with APIs |
| `.prompts/instructions.md` | Global project rules and conventions |
| `.prompts/skills.md` | Reusable agent capabilities catalog |
| `.prompts/prompt-templates.md` | Quick-trigger templates for common tasks |

## When a user asks to implement a feature:
1. Read `.prompts/0-sdlc-orchestrator.md` for the full pipeline
2. Follow the 4-agent sequence: Analysis → Code → Test → UI
3. Apply all rules from `.prompts/instructions.md`
4. Use skills from `.prompts/skills.md` as needed

## Mandatory Code Rules
- **Python**: Logger in every file, type hints, docstrings, Pydantic models, try/except
- **React**: Functional components, custom hooks for data, Tailwind CSS, loading/error/empty states
- **API**: REST conventions, proper status codes, `{"error": "CODE", "message": "..."}` format
- **Git**: Conventional commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)

## Performance Targets
- Face swap: < 50ms/frame
- WebRTC latency: < 150ms
- API endpoints: < 200ms (non-GPU)

## Critical — Do NOT
- Use bare `except:` — always catch specific exceptions
- Put API calls in React components — use custom hooks
- Commit secrets/API keys — use environment variables
- Use inline styles — use Tailwind CSS classes
- Create new CosmosClient per request — reuse singleton
