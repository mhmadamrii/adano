import os

import numpy as np
import pandas as pd
import pytest

# must be set before fraudless.api.* is imported
os.environ["DATABASE_URL"] = "postgresql+psycopg:///fraudless_test"


def synthetic_training_frame(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Test-only synthetic data (NOT a fraud dataset): fraud is likelier for big transfer/cash_out amounts."""
    rng = np.random.default_rng(seed)
    method = rng.choice(["payment", "transfer", "cash_out", "cash_in"], size=n, p=[0.4, 0.2, 0.3, 0.1])
    amount = np.round(np.exp(rng.normal(8, 1.5, size=n)), 2)
    ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 24 * 30, size=n), unit="h")
    p = np.where(np.isin(method, ["transfer", "cash_out"]), np.clip((np.log(amount) - 8) / 8, 0.01, 0.9), 0.002)
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({"amount": amount, "transaction_timestamp": ts, "payment_method": method})
    return df, pd.Series(y, name="label_is_fraud")


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    from fraudless.features import DERIVED_FEATURES, MODEL_INPUT_FIELDS
    from fraudless.modeling import build_pipeline, save_artifact
    from fraudless.schema import SCHEMA_VERSION
    from fraudless.validate import normalize

    X, y = synthetic_training_frame()
    X = X.assign(label_is_fraud=y)
    res = normalize(X, training=True)
    pipe = build_pipeline("logreg").fit(res.clean, res.label)
    meta = {
        "model_version": "test-v0",
        "schema_version": SCHEMA_VERSION,
        "trained_at": "2024-01-01T00:00:00+00:00",
        "threshold": 0.05,
        "model_type": "logreg",
        "metrics": {},
        "features": {"canonical_inputs": list(MODEL_INPUT_FIELDS), "derived": list(DERIVED_FEATURES), "label_used_as_input": False},
        "dataset": {"name": "synthetic test data", "license": "n/a", "label_definition": "test only"},
        "risk_score_interpretation": "test",
    }
    return save_artifact(tmp_path_factory.mktemp("model") / "test-v0", pipe, meta, {})
