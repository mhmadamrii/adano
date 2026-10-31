# Data contract: `transaction-schema-v1`

Source of truth: `fraudless/schema.py` (fields) and `fraudless/validate.py` (rules). Tests: `tests/test_validate.py`.

| Field | Type | Requirement | Rule |
| --- | --- | --- | --- |
| `transaction_id` | string | optional | Generated as `row-<n>` if absent or blank. Duplicates produce a warning, not a rejection. |
| `amount` | number | **required** | Finite, `>= 0`. Blank/invalid/negative rejects the row. |
| `transaction_timestamp` | datetime | recommended | ISO-8601 (`format="ISO8601"`). Naive values are interpreted as **UTC** (warning). Offsets are converted to UTC. Blank = unknown (warning). Unparseable rejects the row. |
| `merchant_category`, `payment_method`, `country` | category | optional | Trimmed, lower-cased, runs of spaces/`-`/`/` become `_`. Blank = unknown. No ISO country mapping in v1. |
| `customer_id` | string | optional | Pseudonymous token only. Not used by the v1 model. |
| `account_age_days`, `avg_transaction_amount` | number | optional | Finite, `>= 0`. |
| `transactions_last_24h` | integer | optional | Non-negative integer. |
| `label_is_fraud` | 0/1 | **training only** | 0 = legitimate, 1 = confirmed fraud. At inference it is ignored (warning) and never reaches the model. |

## Policies

- **No silent drops, no invented values.** Invalid rows are rejected and listed with `row_id` + field + code (cell values are not echoed). Blank optional values stay unknown and are counted in warnings.
- **Unknown is not legitimate.** A missing `payment_method` is encoded as an explicit `unknown` level; an unseen level encodes the same way. Each scored row lists the model inputs that were unknown (`missing_model_inputs`).
- **Explicit column mapping, no header guessing.** A header matches a canonical field if it equals the field name after case/punctuation normalization (`Payment Method` -> `payment_method`). Anything else needs the `mapping` form field: `{"amount": "Amt", "payment_method": "Method"}`. Unknown canonical keys, absent columns, or two fields on one column are 400s.
- **Sensitive columns are refused.** Headers that look like card numbers, CVV/CVC, SSN, passwords, PINs, passports or national IDs reject the whole file (400 `sensitive_column`).
- **Limits.** UTF-8 CSV, comma-separated, `.csv` only. Defaults: 5 MiB and 50,000 rows (`FRAUDLESS_MAX_UPLOAD_BYTES`, `FRAUDLESS_MAX_ROWS`).

## Model inputs in v1

The v1 model reads only `amount`, `transaction_timestamp` (hour-of-day) and `payment_method`. Other canonical fields are accepted and validated but **do not affect the score** because the public training data has no equivalent columns.

## Response contract (`POST /predict/file`)

Enforced by `fraudless/api/schemas.py` and `tests/test_api.py`.

```json
{
  "run_id": "uuid | null",
  "model_version": "v1",
  "schema_version": "transaction-schema-v1",
  "total_rows": 5, "scored_rows": 5, "rejected_row_count": 0, "flagged_rows": 2,
  "threshold": 0.0558,
  "results": [{"row_id": 2, "transaction_id": "t-002", "prediction": "potentially_suspicious", "risk_score": 0.746212, "missing_model_inputs": []}],
  "rejected_rows": [{"row_id": 3, "errors": [{"field": "amount", "code": "invalid_value", "message": "amount must be a finite number"}]}],
  "rejected_rows_truncated": false,
  "warnings": [{"code": "missing_values", "message": "...", "fields": ["merchant_category"], "count": 3, "row_ids": null}],
  "explanation": {"method": "none", "model_type": "...", "features_used": ["amount", "..."], "score_interpretation": "...", "note": "..."},
  "disclaimer": "A flag means 'potentially suspicious' ..."
}
```

- `row_id` is the 1-based position among CSV data rows. `prediction` is `potentially_suspicious` iff `risk_score >= threshold`.
- `risk_score` is a model-estimated probability under the **training prevalence (about 0.13%)**. It is not a certainty and not transferable to another population; scores far below 0.5 are normal.
- `rejected_rows` is capped at 1000 entries (`rejected_rows_truncated`); the count is always exact.

Errors always use `{"error": {"code", "message", "details"}}`:

| Status | Codes |
| --- | --- |
| 400 | `empty_file`, `no_data_rows`, `malformed_csv`, `not_text`, `invalid_encoding`, `unsupported_delimiter`, `duplicate_headers`, `sensitive_column`, `invalid_mapping` |
| 413 | `file_too_large`, `too_many_rows` |
| 415 | `unsupported_file_type` |
| 422 | `missing_required_columns`, `no_valid_rows`, `invalid_request` |
| 500 | `internal_error` (generic; no trace, no row data) |

## Storage (Postgres)

Uploads are processed in memory and never written to disk or the database. Postgres holds:

- `model_versions`: one row per artifact the service has loaded (metadata + sha256).
- `audit_runs`: one row per upload with **aggregates only** (status, error code, counts, warning codes, file sha256/size, duration). No filenames, cells, row ids, or scores. Tested in `test_runs_store_aggregates_only`.

If the database is unreachable the service still scores, `GET /health` reports `degraded`, and the response carries `run_not_recorded`.
