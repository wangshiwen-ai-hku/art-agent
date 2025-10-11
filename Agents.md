# Repository Guidelines

## Project Structure & Module Organization
Backend code sits in `src/`: `service/canvas_agent/` exposes the FastAPI endpoints, `agents/` defines reusable roles, and `infra/tools/` contains the SVG, math, and editing helpers they invoke. Configuration lives in `src/config/`, and shared utilities in `src/utils/`. The React/Vite client lives in `frontend/` (`frontend/src/` for views, `frontend/public/` for static assets). Long-form notes stay in `docs/`, while rendered SVG experiments are archived in `output/` for easy visual diffing.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` — install backend dependencies; append `-e .[dev]` for lint/test extras.
- `uvicorn src.service:app --reload --port 8001` — run the API with auto-reload.
- `npm install` then `npm run dev` inside `frontend/` — launch the chat/generate/edit UI.
- `npm run build` to produce static assets; `npm run lint` to enforce TypeScript lint rules.

## Coding Style & Naming Conventions
Target Python 3.12 with `black` (88-char lines) and `mypy` (typed public APIs). Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and keep tool modules grouped by capability under `src/infra/tools/`. In the frontend, prefer functional React components, `camelCase` helpers, `PascalCase` components, and align with the bundled ESLint rules. Store generated SVGs in timestamped folders beneath `output/`.

## Testing Guidelines
Run backend checks with `pytest`; mirror the source layout when creating suites (`tests/infra/tools/test_svg_tools.py`, etc.) and use markers or `-k` filters to exercise canvas flows. Document integration experiments in `examples/` so others can replay them. Frontend updates must pass `npm run lint`; add vitest or Playwright jobs when you automate UI flows and record the command in your PR description.

## Commit & Pull Request Guidelines
Keep commit messages short, present-tense, and task-focused (`add svg path validator`, `refine canvas routing`). Separate backend and frontend work unless the change is inseparable. Pull requests should describe the user-facing impact, cite new commands or configs, and attach before/after screenshots or SVG snippets for UI or rendering tweaks. Tag relevant reviewers and track any follow-up items in a checklist.

## Agent & Tooling Notes
Register new agent roles under `src/agents/` and expose capabilities through `src/infra/tools/` so the `canvas_agent` router can compose them. Capture bespoke shortcuts or heuristics in `docs/` and reference them from PRs to accelerate onboarding.
