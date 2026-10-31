"""Pipeline construction + artifact IO. Preprocessing and model are one fitted Pipeline."""
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor
from .schema import SCHEMA_VERSION

MODEL_KINDS = ("logreg", "hgb_calibrated")


def build_pipeline(kind: str, seed: int = 42) -> Pipeline:
    # No class reweighting on purpose: it distorts probabilities. Imbalance is handled by
    # stratified splits, PR-AUC, and an explicit decision threshold chosen for recall.
    if kind == "logreg":
        clf = LogisticRegression(max_iter=1000)
    elif kind == "hgb_calibrated":
        clf = CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed), method="isotonic", cv=3
        )
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact(out_dir: Path, pipeline: Pipeline, metadata: dict, evaluation: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / "model.joblib")
    metadata = {
        **metadata,
        "artifact_sha256": sha256_file(out_dir / "model.joblib"),
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (out_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    return out_dir


class ArtifactError(RuntimeError):
    pass


def load_artifact(model_dir: Path) -> tuple[Pipeline, dict]:
    """Load a trusted artifact directory. joblib is pickle-based: only load directories you produced."""
    meta_path, model_path = model_dir / "metadata.json", model_dir / "model.joblib"
    if not meta_path.exists() or not model_path.exists():
        raise ArtifactError(f"no model artifact in {model_dir} (expected metadata.json + model.joblib); run the training command")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactError(f"artifact schema {meta.get('schema_version')!r} != service schema {SCHEMA_VERSION!r}")
    if sha256_file(model_path) != meta.get("artifact_sha256"):
        raise ArtifactError("model.joblib does not match artifact_sha256 in metadata.json (corrupt or modified)")
    return joblib.load(model_path), meta
