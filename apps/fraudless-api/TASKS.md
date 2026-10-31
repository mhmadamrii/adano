# TASKS: Phase 1 (evidence from the verification run)

Phase 2+ not started. No frontend exists.

| # | Task | Status | Evidence |
| --- | --- | --- | --- |
| 0 | Repo reconnaissance | done | Monorepo `ai-apps`: pnpm workspace `apps/*`; flat pairs `apps/<name>` + `apps/<name>-api`; no CLAUDE.md in repo; Python 3.12 venv per API (torch/py3.14 constraints from adano); Podman 5.7 machine; no CI. Only shared files touched: root `package.json` (4 scripts), root `README.md` (1 table row). `adano`, `adano-api` unchanged. |
| 1 | Canonical schema + validation tests | done | `docs/DATA_CONTRACT.md`; `tests/test_validate.py` 33 tests (missing column, invalid/negative/non-finite numbers, empty file, malformed CSV, binary, encoding, delimiter, row limit, bad types, mapping, sensitive headers) |
| 2 | Dataset notes + reproducible training | done | `docs/DATASET.md`; `pnpm train:fraudless`; two runs gave identical `evaluation.json` and artifact sha256 `c5add3fb41db...` |
| 3 | Evaluation + saved artifact | done | `docs/EVALUATION.md`; `models/v1/{model.joblib,metadata.json,evaluation.json}`; test PR-AUC 0.293 vs 0.0013 random, precision 0.363, recall 0.323 |
| 4 | FastAPI upload/inference + tests | done | `tests/test_api.py` (contract keys, label ignored, mapping, rejected rows reported, 13 error cases with 4xx, no trace on 500, Postgres aggregates only, degraded DB) |
| 5 | Podman container + e2e | done | `podman build` ok; container run with `-v models/v1:/models/v1:ro,Z`; `/health` = ok/db ok; curl uploads returned documented JSON; 415/422/400/400 for bad files; runs visible in Postgres; no-model start fails with `no model artifact in /models/v1` |
| 6 | README + completion report | done | `README.md` |

## Exit criteria

1. Training completes reproducibly from documented data: **pass** (identical metrics + sha across runs).
2. Pipeline saved and loadable in a fresh process: **pass** (`tests/test_model.py::test_artifact_loads_in_fresh_process`, `python -I` subprocess).
3. FastAPI starts and loads the artifact: **pass** (host and container).
4. CSV upload returns a valid prediction response: **pass** (5-row sample: 2 flagged).
5. Invalid files get clear 4xx: **pass** (tests + curl against the container).
6. Automated tests pass: **pass**, `pnpm test:fraudless-api` -> `61 passed`.
7. Runs in a Podman container: **pass**.
8. No frontend built: **pass**.

## Known risks / open items

- Model is weak by design (3 inputs, simulated data): 32% recall on test. Balance columns would help; they are not in the v1 contract (Phase 2 candidate).
- Dataset license is ambiguous (CC-BY-4.0 declared vs CC BY-SA 4.0 upstream); re-check before commercial use.
- Only 164 test positives, one seed, no confidence intervals.
- No auth/tenancy/rate limiting; no formatter or linter is configured in the repo, so none was run.
- Upload size is enforced from `Content-Length` and after read; a chunked upload without that header is only bounded after the body is received.
