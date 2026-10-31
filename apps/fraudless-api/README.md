# fraudless-api (Phase 1: ML proof of concept, no UI)

Upload a transaction CSV, get per-row **potentially suspicious** flags and risk scores. Plan: `../../workflows/fraud-less.md`.
A flag is a statistical hint from a model trained on **simulated** data, not evidence of fraud.

```
CSV -> size/type/encoding checks -> column mapping -> canonical schema (transaction-schema-v1)
    -> one sklearn Pipeline (features + preprocessing + model, loaded once at startup) -> JSON
```

| Path | Role |
| --- | --- |
| `fraudless/schema.py`, `validate.py`, `csvio.py` | Data contract, mapping, validation, safe CSV parsing (shared) |
| `fraudless/features.py`, `modeling.py` | Features + pipeline + artifact IO (shared by training and serving) |
| `fraudless/training/` | Training CLI, dataset adapter, evaluation (never imported by the API) |
| `fraudless/api/` | FastAPI app, Postgres models |
| `docs/` | `DATA_CONTRACT.md`, `DATASET.md`, `EVALUATION.md` |
| `TASKS.md` | Task checklist with evidence |

## Setup (from the repo root)

Python 3.12 and a running PostgreSQL (`brew services start postgresql@16`).

```sh
createdb fraudless && createdb fraudless_test     # DATABASE_URL overrides postgresql+psycopg:///fraudless
cd apps/fraudless-api
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
cd ../..
```

## Train, test, run

```sh
pnpm train:fraudless          # downloads the pinned dataset (28 MB, sha256-checked), writes apps/fraudless-api/models/v1/
pnpm test:fraudless-api       # 61 tests; they train a tiny model on synthetic data and use the fraudless_test database
pnpm dev:fraudless-api        # http://localhost:8001  (docs at /docs)
```

```sh
curl -s localhost:8001/health
curl -s localhost:8001/model-info
curl -s -F "file=@apps/fraudless-api/samples/transactions_sample.csv;type=text/csv" localhost:8001/predict/file
# customer headers differ -> pass a mapping
curl -s -F "file=@apps/fraudless-api/samples/custom_headers.csv;type=text/csv" \
        -F "mapping=<apps/fraudless-api/samples/custom_headers.mapping.json" localhost:8001/predict/file
curl -s localhost:8001/runs     # aggregate audit-run history from Postgres
```

Contract, error codes and policies: `docs/DATA_CONTRACT.md`. Dataset, license and limits: `docs/DATASET.md`. Metrics: `docs/EVALUATION.md`.

## Podman

The image holds code only; the model artifact is mounted read-only.

```sh
pnpm image:fraudless-api
podman run --rm -p 8001:8001 \
  -v "$PWD/apps/fraudless-api/models/v1:/models/v1:ro,Z" \
  -e DATABASE_URL=postgresql+psycopg://$USER@host.containers.internal:5432/fraudless \
  fraudless-api:phase1
```

`host.containers.internal` reaches the Homebrew Postgres on the host (it listens on localhost and trusts local connections by default). With no model mounted the container exits at startup with `no model artifact in /models/v1`. Limits are env vars: `FRAUDLESS_MODEL_DIR`, `FRAUDLESS_MAX_UPLOAD_BYTES`, `FRAUDLESS_MAX_ROWS`.

## Security notes

- Uploads are untrusted: size, row, type, encoding and header checks; processed in memory; never stored. Raw rows are never logged (errors log only the exception type).
- `model.joblib` is a pickle: only mount artifacts you trained. The loader checks the SHA-256 in `metadata.json` to catch corruption, which does not stop a malicious pickle.
- No authentication, tenancy or rate limiting yet (Phase 4). CORS is not enabled.
