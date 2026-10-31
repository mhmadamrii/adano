"""Dataset acquisition + adapter: PaySim partition -> canonical transaction-schema-v1 frame.

Only a download + checksum + column adapter lives here; the data itself is not committed (see docs/DATASET.md).
"""
import urllib.request
from pathlib import Path

import pandas as pd

from ..modeling import sha256_file

HF_REPO = "flwrlabs/fed-fraud-paysim-banks"
HF_REVISION = "a48633710ded3045aa9c80833498ab16de3560ca"
HF_FILE = "data/test-00000-of-00001.parquet"
SHA256 = "2baf849329b68e9ad293e8aeab1e01c2625bde2a3f1c26e0dc55111339a6c27a"
URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{HF_FILE}"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "paysim-partition" / "test-00000-of-00001.parquet"

# PaySim simulates 744 hourly steps; the anchor date is arbitrary and only fixes hour-of-day = (step-1) % 24.
SIMULATION_ANCHOR = pd.Timestamp("2000-01-01T00:00:00Z")

DATASET_INFO = {
    "name": "PaySim-style synthetic mobile-money transactions (partition used: test split of flwrlabs/fed-fraud-paysim-banks)",
    "source_url": f"https://huggingface.co/datasets/{HF_REPO}",
    "revision": HF_REVISION,
    "file": HF_FILE,
    "file_sha256": SHA256,
    "license": "cc-by-4.0 as declared by the hosting repo; the original PaySim release (Lopez-Rojas et al., 2016) is listed on Kaggle as CC BY-SA 4.0 - treat the stricter terms as applicable",
    "label_definition": "isFraud = 1 for fraudulent transactions injected by the PaySim simulator; these are SIMULATED labels, not confirmed real-world fraud outcomes",
    "field_mapping": {
        "amount": "amount",
        "payment_method": "type (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER)",
        "transaction_timestamp": "SIMULATION_ANCHOR + (step-1) hours",
        "customer_id": "nameOrig (pseudonymous account id)",
        "label_is_fraud": "isFraud",
    },
    "unused_columns": ["oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud", "BankID"],
}


def download(path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    return path


def verify(path: Path) -> None:
    got = sha256_file(path)
    if got != SHA256:
        raise RuntimeError(f"{path} sha256 {got} != pinned {SHA256}; delete it to re-download")


def load_canonical(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Canonical columns (incl. label_is_fraud) ready for validate.normalize(training=True)."""
    df = pd.read_parquet(path)
    return pd.DataFrame(
        {
            "amount": df["amount"].astype("float64"),
            "transaction_timestamp": SIMULATION_ANCHOR + pd.to_timedelta(df["step"].astype("int64") - 1, unit="h"),
            "payment_method": df["type"].astype(object),
            "customer_id": df["nameOrig"].astype(object),
            "label_is_fraud": df["isFraud"].astype("int64"),
        }
    )
