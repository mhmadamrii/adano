"""Canonical frame -> model features. Shared by training and serving so they cannot drift."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Canonical fields the v1 model reads. The label is deliberately absent: it can never be an input.
MODEL_INPUT_FIELDS = ("amount", "transaction_timestamp", "payment_method")
DERIVED_FEATURES = ("log_amount", "hour_sin", "hour_cos", "payment_method")
NUMERIC = ["log_amount", "hour_sin", "hour_cos"]
CATEGORICAL = ["payment_method"]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=list(MODEL_INPUT_FIELDS))
    ts = pd.to_datetime(df["transaction_timestamp"], utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0  # NaN where timestamp is unknown
    angle = 2 * np.pi * hour / 24.0
    return pd.DataFrame(
        {
            "log_amount": np.log1p(df["amount"].astype("float64")),
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
            "payment_method": df["payment_method"].astype(object),
        },
        index=df.index,
    )


def missing_inputs(df: pd.DataFrame) -> pd.Series:
    """Per row: which model inputs were unknown (explanation metadata, not a score input)."""
    d = df.reindex(columns=list(MODEL_INPUT_FIELDS))
    return d.isna().apply(lambda r: [c for c in MODEL_INPUT_FIELDS if r[c]], axis=1)


def build_preprocessor() -> Pipeline:
    column_tf = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), NUMERIC),
            (
                "cat",
                Pipeline(
                    [
                        # unknown is an explicit level, never "legitimate"; unseen levels encode to all-zeros
                        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )
    return Pipeline([("derive", FunctionTransformer(derive, validate=False)), ("columns", column_tf)])
