# FraudLess --- Project Goal & Implementation Plan

## 1. Product Goal

Build an AI-powered transaction audit platform inside the existing
monorepo. Companies will eventually be able to upload transaction files,
have the system validate and normalize their data, run fraud/anomaly
inference, and review explainable risk results.

**Primary objective for Phase 1:** prove that the end-to-end ML workflow
works --- from training a baseline model to accepting an uploaded CSV
and returning inference results through an API. Do not build a frontend
or polished UI in this phase.

This is a proof of concept, not a production fraud-decision system. A
model flag means _potentially suspicious_ and must not be presented as
proof of fraud.

## 2. Existing Repository & Technology Constraints

- This work belongs inside the **existing monorepo**. Do not create a
  separate repository or scatter related projects elsewhere.
- First inspect the repository structure, existing conventions,
  package management, shared libraries, and the existing `adano` / AI
  projects.
- Avoid modifying unrelated applications or shared configuration
  unless required and justified.
- Frontend stack for later phases: **React, TanStack Start,
  TypeScript, Tailwind CSS, shadcn/ui**.
- Backend / ML API: **Python + FastAPI**.
- Container runtime: **Podman**. Prefer OCI-compatible Containerfiles
  and commands that work with Podman.
- Keep the ML service independently runnable within the monorepo, with
  clear boundaries and minimal coupling to frontend code.
- Do not introduce extra infrastructure or dependencies without a
  concrete Phase 1 need.

## 3. Core Data Flow

```text
Transaction CSV upload
        |
        v
File validation
        |
        v
Column mapping / Data Contract
        |
        v
Canonical transaction schema
        |
        v
Preprocessing pipeline
        |
        v
Trained ML model inference
        |
        v
Risk score + prediction + explanation metadata
        |
        v
JSON response
```

The uploaded file's column names may differ between companies. The
backend must normalize supported input formats into one internal
canonical schema before inference.

## 4. Data Contract

Define and document a versioned canonical schema. Start with a small,
defensible set of fields and make optional fields explicit.

Suggested initial fields:

---

Canonical field Type Required Notes

---

`transaction_id` string No Generate an
internal ID if
absent

`amount` number Yes Transaction
amount; validate
finite and
non-negative

`transaction_timestamp` datetime/string Recommended Parse with
timezone
handling
documented

`merchant_category` string No Normalize
categories

`payment_method` string No Normalize
categories

`country` string No Country
code/name
normalization

`customer_id` string No Pseudonymous
identifier; do
not require
direct identity

`account_age_days` number No Must be
non-negative

`transactions_last_24h` integer No Must be
non-negative

`avg_transaction_amount` number No Must be finite
and non-negative

`label_is_fraud` integer Training only 0 = legitimate,
1 = confirmed
fraud

---

Important rules:

- Separate **training data** (which may contain a confirmed label)
  from **inference data** (which must not require or use the label).
- Never silently invent missing values or treat unknown values as
  legitimate.
- Return clear validation errors identifying invalid/missing columns
  and rows.
- Support an explicit column-mapping configuration so customer CSV
  headers can map to canonical fields.
- Version the schema, for example `transaction-schema-v1`.
- Do not assume every optional field exists. The model's preprocessing
  must handle the agreed missing/unknown-value policy consistently.
- Do not include highly sensitive personal data such as full card
  numbers, CVV, passwords, or government identifiers. Use
  tokenized/pseudonymous IDs only.

## 5. Phase Plan

### Phase 0 --- Monorepo Reconnaissance

Before coding:

1.  Inspect the monorepo and identify the correct location for the new
    service.
2.  Read existing project instructions and conventions (including
    `CLAUDE.md`, if present).
3.  Identify Python version/tooling already used in the repository.
4.  Identify existing Podman conventions and CI/test commands.
5.  Propose the smallest implementation plan and list files that will be
    added or changed.
6.  Do not modify unrelated projects.

**Exit criteria:** repository structure and integration plan are
documented; no unrelated files changed.

### Phase 1 --- ML Proof of Concept (No UI)

Build the smallest complete training-to-inference workflow.

#### A. Dataset and training

- Select a public, appropriately licensed transaction-fraud dataset
  and record its source, license, target definition, and limitations.
- Do not fabricate labels or claim that synthetic labels are real
  fraud outcomes.
- Inspect class balance and missing values.
- Split train/test data with stratification where appropriate.
- Prevent data leakage: split before fitting preprocessing; fit
  imputers/encoders/scalers only on training data.
- Build preprocessing and model as one scikit-learn `Pipeline` /
  `ColumnTransformer` where practical.
- Establish a simple baseline model first (for example, Logistic
  Regression or Random Forest); compare a second model only if useful.
- Handle class imbalance deliberately. Do not rely on accuracy alone.
- Evaluate with confusion matrix, precision, recall, F1, and PR-AUC
  where appropriate. Explain the trade-off between false positives and
  false negatives.
- Persist the complete fitted preprocessing + model pipeline as one
  versioned artifact.
- Record model metadata: dataset/source, schema version, feature list,
  training date, model version, split strategy, and evaluation
  metrics.

#### B. Upload and inference API

Implement a minimal FastAPI service with:

- `GET /health` --- service health.
- `GET /model-info` --- loaded model version, expected schema version,
  and supported features.
- `POST /predict/file` --- accept a CSV upload, validate it, map
  columns to the canonical schema, preprocess it, run inference, and
  return per-row results plus a summary.
- A small CLI/script for training; training should not run on every
  API request.
- Load the model once at application startup, not once per request.
- Return stable JSON responses with row identifiers, prediction, risk
  score/probability (only if the model supports a
  calibrated/probabilistic interpretation), and validation warnings.
- Do not claim that a probability is a certainty or that a flagged row
  is confirmed fraud.

Example response shape (illustrative):

```json
{
  "model_version": "baseline-v1",
  "schema_version": "transaction-schema-v1",
  "total_rows": 2,
  "flagged_rows": 1,
  "results": [
    {
      "row_id": 1,
      "prediction": "potentially_suspicious",
      "risk_score": 0.87
    },
    {
      "row_id": 2,
      "prediction": "not_flagged",
      "risk_score": 0.08
    }
  ],
  "warnings": []
}
```

The exact response contract must be documented and tested. Do not expose
raw exception traces to clients.

#### C. Storage decision for Phase 1

**Object storage is not required for the initial proof of concept.**

- Accept the uploaded CSV in FastAPI and process it in a controlled
  temporary directory or stream it where practical.
- Enforce file size and row limits; validate file type and encoding.
- Delete temporary upload files after processing, including on errors.
- Persist the trained model artifact separately on a mounted Podman
  volume.
- Do not store uploaded company data by default in Phase 1.

Create a small storage abstraction only if it is genuinely useful. Keep
the design compatible with adding S3-compatible object storage (for
example, MinIO) in a later phase for durable uploads, reports, and
larger asynchronous jobs. Do not deploy MinIO just to complete Phase 1.

#### D. Tests and documentation

- Unit tests for schema validation, column mapping, preprocessing, and
  response formatting.
- Tests for missing required columns, invalid numeric values, empty
  files, malformed CSV, and unsupported file types.
- An integration test that uploads a small fixture CSV and checks the
  response.
- A reproducible training command and inference run documented in the
  README.
- A sample CSV containing safe, non-sensitive example data.
- A Podman-compatible Containerfile and instructions for mounting the
  model artifact.

**Phase 1 exit criteria:**

1.  Training completes reproducibly from documented data.
2.  The full preprocessing/model pipeline is saved and can be loaded in
    a fresh process.
3.  FastAPI starts and loads the artifact successfully.
4.  A CSV can be uploaded and produces a valid prediction response.
5.  Invalid files receive clear 4xx responses.
6.  Automated tests pass.
7.  The service runs in a Podman container.
8.  No frontend/UI is built.

### Phase 2 --- Data Quality & Audit Experience

- Add a configurable CSV column-mapping flow.
- Add richer data-quality reports and downloadable audit results.
- Add per-row explanations using SHAP or another appropriate
  explainability method.
- Clearly distinguish model explanations from verified evidence.
- Add support for larger files and background processing only when
  needed.

### Phase 3 --- Web Application

Build the frontend using the existing stack:

- React + TanStack Start
- TypeScript
- Tailwind CSS
- shadcn/ui

User flow: sign in → upload file → map columns → review validation → run
audit → inspect dashboard and flagged rows → export report.

The frontend must consume documented FastAPI contracts; do not duplicate
ML preprocessing in TypeScript.

### Phase 4 --- SaaS & Production Hardening

- Authentication, organization/tenant isolation, authorization, and
  audit logs.
- Durable object storage (S3-compatible) for uploads and reports.
- Background jobs, retries, progress reporting, and cancellation.
- Model/version registry, monitoring, drift checks, and retraining
  workflow.
- Retention/deletion policies, encryption, secrets management, and
  operational metrics.
- Human review workflow and feedback labels, with safeguards against
  blindly retraining on unverified predictions.

## 6. Engineering Principles

- Keep the first implementation small and verifiable.
- Prefer explicit schemas, typed interfaces, deterministic
  preprocessing, and reproducible experiments.
- Keep training code separate from serving code.
- Never fit preprocessing on test data.
- Never use the target/label as an inference feature.
- Do not hardcode assumptions about customer CSV headers.
- Do not silently drop rows; report rejected rows and reasons.
- Avoid overclaiming model quality. Public benchmark performance does
  not guarantee performance on a company's real transactions.
- Treat uploaded files as untrusted input. Validate size, type,
  encoding, columns, and values.
- Do not log raw transaction rows or sensitive customer data.
- Ask before adding major dependencies, infrastructure, or changing
  monorepo-wide configuration.

## 7. Claude Code Working Protocol --- Verifiable, RLVR-Inspired

Work in small tasks with explicit, machine-checkable acceptance
criteria. This is an engineering workflow inspired by verifiable-reward
loops; it is **not** a request to train or fine-tune Claude.

For every task:

1.  **Inspect:** read relevant files and repository instructions before
    editing.
2.  **Plan:** state the intended change, files affected, assumptions,
    and acceptance criteria.
3.  **Implement:** make the smallest change that satisfies the task.
4.  **Verify:** run relevant formatters, type checks, unit tests,
    integration tests, and/or a reproducible CLI command.
5.  **Report:** list changed files, exact commands run, pass/fail
    results, and remaining risks.
6.  **Iterate:** if a check fails, diagnose and fix it before moving to
    the next task.

Rules:

- Do not claim success unless the corresponding command/test actually
  passed.
- Do not skip or weaken tests just to obtain a passing result.
- Do not mark a task complete when acceptance criteria are unverified.
- Keep a short `TASKS.md` or equivalent checklist with status and
  evidence.
- Work on one phase/task at a time. Do not start Phase 2 or build UI
  until Phase 1 exit criteria pass.
- If requirements are ambiguous or a destructive/unrelated change
  appears necessary, stop and ask.

Suggested task sequence for Claude Code:

1.  Repository reconnaissance and implementation proposal.
2.  Canonical schema + validation tests.
3.  Dataset acquisition notes + reproducible training pipeline.
4.  Evaluation report + saved model artifact.
5.  FastAPI upload/inference endpoint + tests.
6.  Podman container + end-to-end verification.
7.  Phase 1 README and completion report.

## 8. Definition of Success

Phase 1 is successful when a developer can clone/use the existing
monorepo, follow documented commands, train the baseline model, start
the FastAPI service in Podman, upload a valid CSV, receive documented
per-row inference results, and reproduce the automated test results ---
without any frontend being implemented.
