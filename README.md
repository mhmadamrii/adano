# adano

Adaptive personal note classifier (see `thesis.md`). Monorepo:

- `apps/web` — TanStack Start (React) UI
- `apps/api` — FastAPI + sentence-transformers (hybrid scorer, thesis §3.6)
- PostgreSQL (`categories`, `notes` incl. float32 embeddings as BYTEA); tables auto-created at API startup

## Run

```sh
# once
pnpm install
cd apps/api && python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt   # py3.12; torch has no 3.14 / Intel-mac>2.2 wheels

createdb adano && createdb adano_test   # local socket; override with DATABASE_URL=postgresql+psycopg://user:pw@host/db

# dev (two terminals)
pnpm dev:api   # :8000
pnpm dev:web   # :3000, proxies /api -> :8000

pnpm test:api
```
