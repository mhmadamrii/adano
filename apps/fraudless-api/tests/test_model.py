import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudless.features import MODEL_INPUT_FIELDS, derive
from fraudless.modeling import ArtifactError, build_pipeline, load_artifact
from fraudless.schema import CANONICAL_NAMES
from fraudless.training.evaluate import choose_threshold, evaluate
from fraudless.validate import normalize

from .conftest import synthetic_training_frame


def test_label_is_never_a_model_input():
    assert "label_is_fraud" not in MODEL_INPUT_FIELDS
    assert set(MODEL_INPUT_FIELDS) <= set(CANONICAL_NAMES)
    X, y = synthetic_training_frame(200)
    X = X.assign(label_is_fraud=y)
    assert "label_is_fraud" not in derive(normalize(X, training=True).clean).columns


def test_preprocessing_is_fit_on_training_data_only():
    X, y = synthetic_training_frame(2000)
    res = normalize(X.assign(label_is_fraud=y), training=True)
    train, test = res.clean.iloc[:1500], res.clean.iloc[1500:].copy()
    test["amount"] = test["amount"] * 1000  # shift the test distribution wildly
    pipe = build_pipeline("logreg").fit(train, res.label.iloc[:1500])
    scaler = pipe.named_steps["prep"].named_steps["columns"].named_transformers_["num"].named_steps["scale"]
    expected_mean = np.log1p(train["amount"]).mean()
    pipe.predict_proba(test)  # scoring must not refit anything
    assert scaler.mean_[0] == pytest.approx(expected_mean)


def test_unknown_and_missing_values_follow_policy():
    X, y = synthetic_training_frame(2000)
    res = normalize(X.assign(label_is_fraud=y), training=True)
    pipe = build_pipeline("logreg").fit(res.clean, res.label)
    odd = normalize(pd.DataFrame({"amount": ["1000", "1000", "1000"], "payment_method": ["transfer", "hoverboard", ""], "transaction_timestamp": ["2024-01-01T00:00:00Z", "", ""]}).astype("string")).clean
    p = pipe.predict_proba(odd)[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert p[1] == pytest.approx(p[2])  # unseen category == missing category == explicit "unknown"


def test_threshold_and_metrics_basic():
    y = np.array([0] * 90 + [1] * 10)
    proba = np.concatenate([np.linspace(0, 0.4, 90), np.linspace(0.3, 0.9, 10)])
    thr = choose_threshold(y, proba)
    m = evaluate(y, proba, thr)
    assert m["confusion_matrix"]["tp"] + m["confusion_matrix"]["fn"] == 10
    assert 0 <= m["pr_auc"] <= 1 and m["pr_auc"] > m["pr_auc_random_baseline"]


def test_artifact_loads_in_fresh_process(model_dir):
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        import pandas as pd
        from pathlib import Path
        from fraudless.modeling import load_artifact
        from fraudless.validate import normalize
        pipe, meta = load_artifact(Path({str(model_dir)!r}))
        df = normalize(pd.DataFrame({{"amount": ["9000000"], "payment_method": ["transfer"]}}).astype("string")).clean
        p = float(pipe.predict_proba(df)[0, 1])
        assert 0 <= p <= 1, p
        print("ok", meta["model_version"])
        """
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok test-v0", out.stderr


def test_load_artifact_rejects_missing_tampered_and_wrong_schema(model_dir, tmp_path):
    import json
    import shutil

    with pytest.raises(ArtifactError, match="no model artifact"):
        load_artifact(tmp_path / "nope")
    bad = tmp_path / "tampered"
    shutil.copytree(model_dir, bad)
    (bad / "model.joblib").write_bytes((bad / "model.joblib").read_bytes() + b"x")
    with pytest.raises(ArtifactError, match="sha256"):
        load_artifact(bad)
    wrong = tmp_path / "wrong"
    shutil.copytree(model_dir, wrong)
    meta = json.loads((wrong / "metadata.json").read_text())
    meta["schema_version"] = "transaction-schema-v0"
    (wrong / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ArtifactError, match="schema"):
        load_artifact(wrong)
