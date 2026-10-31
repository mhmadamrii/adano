# ai-apps

Monorepo of AI integration projects. Each project is a pair under `apps/`:
a web app (`apps/<name>`, TanStack Start, pnpm workspace) and, when needed, a
Python backend (`apps/<name>-api`, own venv, not part of the pnpm workspace).

| Project | Web | API | Notes |
| --- | --- | --- | --- |
| fraudless | `apps/fraudless` (minimal upload/review UI) | `apps/fraudless-api` | Transaction audit / fraud-risk scoring. Phase 1 is API + ML only, see `apps/fraudless-api/README.md`. Plan: `workflows/fraud-less.md` |
| adano | `apps/adano` | `apps/adano-api` | Adaptive note classifier, see `apps/adano/thesis.md`. FastAPI + sentence-transformers + PostgreSQL (`adano` db) |

## adano

```sh
pnpm install
createdb adano && createdb adano_test   # local socket; override with DATABASE_URL=postgresql+psycopg://user:pw@host/db

# Python 3.12 (torch has no 3.14 wheels; Intel mac caps at torch 2.2.2, hence the pins)
cd apps/adano-api && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ../..

pnpm dev:adano-api   # :8000
pnpm dev:adano       # :3000, proxies /api -> :8000
pnpm test:adano-api
```

Adding a project: `apps/<name>` (+ `apps/<name>-api`), then add `dev:<name>` scripts at the root. Pick a free API port.

## Shared dependency versions (pnpm catalog)

`pnpm-workspace.yaml` has a `catalog:` block holding the version of every shared web dependency (TanStack Start/Router, React, Vite, TypeScript, Tailwind, ...). Apps declare them as `"catalog:"` in their `package.json`, so a bump happens in one place and every app moves together. A new web app copies `apps/fraudless/package.json` and `vite.config.ts`; add a dependency to the catalog only when two apps need it.

## fraudless

```sh
pnpm dev:fraudless-api   # :8001 (needs Postgres, trained model: pnpm train:fraudless)
pnpm dev:fraudless       # :3001, proxies /api -> :8001
```
