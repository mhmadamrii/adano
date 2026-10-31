# Dataset notes

| | |
| --- | --- |
| Data | PaySim-style **synthetic** mobile-money transactions, 636,262 rows |
| Used partition | `data/test-00000-of-00001.parquet` of Hugging Face `flwrlabs/fed-fraud-paysim-banks` |
| Pinned revision | `a48633710ded3045aa9c80833498ab16de3560ca` |
| File SHA-256 | `2baf849329b68e9ad293e8aeab1e01c2625bde2a3f1c26e0dc55111339a6c27a` (checked before every training run) |
| License | The hosting repo declares **CC-BY-4.0**. That repo derives from a mirror of PaySim (Lopez-Rojas, Elmir, Axelsson, EXPO 2016), which is listed on Kaggle as **CC BY-SA 4.0**. The declared licenses disagree, so assume the stricter share-alike terms for anything derived from the data, and re-check before any commercial use. |
| Not committed | The parquet is downloaded by the training CLI into `data/raw/` (git-ignored). |

**Target.** `isFraud = 1` marks fraud *injected by the simulator*. These are simulated labels, not confirmed real-world fraud outcomes, and nothing here should be read as real fraud performance.

**Profile (from the training run).** 636,262 rows, 821 positives (0.129%), 0 nulls, 0 duplicate rows, 636,162 distinct `nameOrig`. All fraud sits in two types: `TRANSFER` 427 of 53,584 and `CASH_OUT` 394 of 223,617. `CASH_IN`, `DEBIT` and `PAYMENT` have none.

**Mapping to the canonical schema** (`fraudless/training/dataset.py`): `amount`, `type` -> `payment_method`, `step` -> `transaction_timestamp` (anchor `2000-01-01T00:00Z` + `step-1` hours; the date is arbitrary and only fixes hour-of-day), `nameOrig` -> `customer_id`, `isFraud` -> `label_is_fraud`. Unused: balances (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), `nameDest`, `isFlaggedFraud`, `BankID`. The balance columns carry most of PaySim's real signal, so the model is deliberately weaker than what the dataset allows; they are not in the v1 contract.

**Limitations.**
- Simulated behaviour: fraud follows the simulator's rules (large transfer/cash-out), which real fraudsters need not follow.
- Only 3 inputs reach the model; `customer_id` is almost unique per row, so no customer-history features exist.
- Random row split: no temporal or customer-grouped validation, so drift over time is not measured.
- A public benchmark number says nothing about a company's real transactions.
