"""Training CLI (never invoked by the API).

    python -m fraudless.training.train --version v1
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from ..features import DERIVED_FEATURES, MODEL_INPUT_FIELDS
from ..modeling import MODEL_KINDS, build_pipeline, save_artifact
from ..schema import SCHEMA_VERSION
from ..validate import normalize
from . import dataset
from .evaluate import THRESHOLD_BETA, choose_threshold, evaluate

SEED = 42
TEST_SIZE = 0.2
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "models"


def run(version: str, data_path: Path, out_root: Path, *, seed: int = SEED, kinds=MODEL_KINDS, log=print) -> Path:
    raw = dataset.load_canonical(data_path)
    res = normalize(raw, training=True)  # same validator as serving
    if res.rejected:
        log(f"training rows rejected by validation: {len(res.rejected)} (reported, not silently dropped)")
    X, y = res.clean, res.label
    log(f"rows={len(X)} fraud={int(y.sum())} ({y.mean():.4%})")

    # split BEFORE anything is fitted; stratified because of the extreme imbalance
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    log(f"train={len(X_tr)} (fraud {int(y_tr.sum())})  test={len(X_te)} (fraud {int(y_te.sum())})")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    candidates: dict[str, dict] = {}
    for kind in kinds:
        # out-of-fold predictions on TRAIN only: preprocessing is re-fit inside every fold
        oof = cross_val_predict(build_pipeline(kind, seed), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = choose_threshold(y_tr, oof)
        candidates[kind] = {"oof": evaluate(y_tr, oof, thr), "threshold": thr}
        log(f"[{kind}] OOF PR-AUC={candidates[kind]['oof']['pr_auc']:.4f} recall={candidates[kind]['oof']['recall']:.3f} precision={candidates[kind]['oof']['precision']:.3f} thr={thr:.5f}")

    # selection uses train-only evidence; the test set is scored once, for the winner
    best = max(candidates, key=lambda k: candidates[k]["oof"]["pr_auc"])
    threshold = candidates[best]["threshold"]
    pipeline = build_pipeline(best, seed).fit(X_tr, y_tr)
    test_metrics = evaluate(y_te, pipeline.predict_proba(X_te)[:, 1], threshold)
    log(f"[selected {best}] TEST PR-AUC={test_metrics['pr_auc']:.4f} (random={test_metrics['pr_auc_random_baseline']:.4f}) "
        f"precision={test_metrics['precision']:.3f} recall={test_metrics['recall']:.3f} cm={test_metrics['confusion_matrix']}")

    metadata = {
        "model_version": version,
        "schema_version": SCHEMA_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_type": best,
        "dataset": dataset.DATASET_INFO,
        "features": {"canonical_inputs": list(MODEL_INPUT_FIELDS), "derived": list(DERIVED_FEATURES), "label_used_as_input": False},
        "split": {
            "strategy": "stratified random row split on isFraud (no temporal or customer grouping)",
            "test_size": TEST_SIZE,
            "seed": seed,
            "train_rows": int(len(X_tr)),
            "test_rows": int(len(X_te)),
            "train_positives": int(y_tr.sum()),
            "test_positives": int(y_te.sum()),
            "selection": "5-fold stratified out-of-fold PR-AUC on the training split only",
        },
        "threshold": threshold,
        "threshold_strategy": f"maximize F{THRESHOLD_BETA:g} on out-of-fold training predictions",
        "metrics": {"test": test_metrics},
        "risk_score_interpretation": "model-estimated probability under the training prevalence (calibrated or near-calibrated); NOT a confirmed-fraud probability for any other population",
    }
    evaluation = {"candidates_oof_train": candidates, "selected": best, "test": test_metrics}
    return save_artifact(out_root / version, pipeline, metadata, evaluation)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--data", type=Path, default=dataset.DEFAULT_PATH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-download", action="store_true", help="fail instead of downloading the pinned dataset")
    args = ap.parse_args(argv)
    if not args.data.exists():
        if args.no_download or args.data != dataset.DEFAULT_PATH:
            print(f"dataset not found: {args.data}", file=sys.stderr)
            return 2
        print(f"downloading pinned dataset -> {args.data}")
        dataset.download(args.data)
    if args.data == dataset.DEFAULT_PATH:
        dataset.verify(args.data)
    out = run(args.version, args.data, args.out)
    print(f"artifact written: {out}")
    print(json.dumps(json.loads((out / "evaluation.json").read_text())["test"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
