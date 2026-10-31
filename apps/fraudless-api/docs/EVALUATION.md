# Evaluation report: model `v1`

Reproduce: `pnpm train:fraudless` (about 30 s on 4 cores). Two consecutive runs produced byte-identical `evaluation.json` and the same artifact SHA-256 (seed 42). Raw numbers: `models/v1/evaluation.json`.

## Protocol

- 80/20 **stratified** split on the label *before* anything is fitted: 509,009 train (657 fraud) / 127,253 test (164 fraud).
- Preprocessing (median imputer, scaler, one-hot encoder) lives inside one scikit-learn `Pipeline` and is re-fit inside every cross-validation fold and on the training split only.
- Two candidates compared with 5-fold stratified **out-of-fold** predictions on the training split: logistic regression (baseline) and calibrated histogram gradient boosting. Selection and the flag threshold use only those predictions. The test split is scored once, for the winner.
- Imbalance: no class reweighting (it distorts probabilities). Handled by stratification, PR-AUC, and a decision threshold chosen to maximize F2 (recall weighted 2x) on out-of-fold training predictions.

## Results

| Split | Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Train, out-of-fold | logreg (baseline) | 0.136 | 0.895 | 0.137 | 0.279 | 0.183 |
| Train, out-of-fold | hgb_calibrated | **0.256** | 0.921 | 0.336 | 0.314 | 0.324 |
| **Test (selected: hgb_calibrated)** | | **0.293** | 0.925 | 0.363 | 0.323 | 0.342 |

Random-guess PR-AUC equals the prevalence, 0.0013, so the model carries real signal, but 0.29 is modest.

Test confusion matrix at threshold 0.0558: TN 126,996, FP 93, FN 111, TP 53. About 1.1 of every 1,000 transactions are flagged. Mean predicted probability 0.00127 vs prevalence 0.00129; Brier 0.00106.

## Trade-offs

- **False negatives** (111 missed frauds, 68% of fraud) are lost money. **False positives** (93 legitimate transactions flagged) cost reviewer time and customer friction. The F2 threshold leans toward recall, yet recall is still only 32%.
- Lowering the threshold catches more fraud and floods reviewers; raising it does the opposite. The threshold is a business decision and is exposed in `/model-info` and every response.
- Accuracy is meaningless here: predicting "never fraud" scores 99.87%.

## Honest caveats

Simulated data and labels, three input features, random split, one seed. Only 164 test positives, so each metric has wide uncertainty (no confidence intervals computed). These numbers do not predict real-world performance.
