import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    max_upload_bytes: int
    max_rows: int

    @classmethod
    def from_env(cls) -> "Settings":
        default_model = Path(__file__).resolve().parents[2] / "models" / "v1"
        return cls(
            model_dir=Path(os.environ.get("FRAUDLESS_MODEL_DIR", default_model)),
            max_upload_bytes=int(os.environ.get("FRAUDLESS_MAX_UPLOAD_BYTES", 5 * 1024 * 1024)),
            max_rows=int(os.environ.get("FRAUDLESS_MAX_ROWS", 50_000)),
        )
