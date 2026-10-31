import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

THRESHOLD_BETA = 2.0  # recall counts 2x as much as precision when picking the flag threshold


def choose_threshold(y_true, proba) -> float:
    """Maximize F-beta (beta=2) on out-of-fold training predictions. Never touches the test set."""
    p, r, t = precision_recall_curve(y_true, proba)
    p, r = p[:-1], r[:-1]
    b2 = THRESHOLD_BETA**2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((p + r) > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    return float(t[int(np.argmax(f))])


def evaluate(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "pr_auc_random_baseline": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "flagged_per_1000": float(pred.mean() * 1000),
        "brier": float(brier_score_loss(y_true, proba)),
        "mean_predicted_probability": float(np.mean(proba)),
    }
