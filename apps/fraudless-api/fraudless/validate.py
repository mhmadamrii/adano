"""Column mapping + row validation: raw tabular input -> canonical schema (transaction-schema-v1).

Policies (documented in docs/DATA_CONTRACT.md):
- A row with a present-but-invalid value, or a missing required value, is rejected and reported; never dropped silently.
- A blank optional/recommended value is kept as missing (NaN) and reported as a warning; nothing is invented here.
- Header matching is explicit: exact (case/punctuation-insensitive) canonical name, or the caller's mapping.
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import CANONICAL_NAMES, FIELD_BY_NAME, FIELDS, RECOMMENDED, REQUIRED

MAX_LISTED_ROWS = 20

_SENSITIVE_HEADER = re.compile(
    r"(^|_)(cvv|cvc|ssn|password|passwd|pin)($|_)|card_?(number|no|num)$|^pan$|passport|national_?id|^nik$"
)
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$", re.IGNORECASE)


class FileValidationError(Exception):
    """File-level problem: nothing can be scored. Maps to a 4xx response."""

    def __init__(self, code: str, message: str, details: dict | None = None, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.details, self.status = code, message, details or {}, status


@dataclass
class RowIssue:
    row_id: int
    errors: list[dict]


@dataclass
class Warning_:
    code: str
    message: str
    fields: list[str] | None = None
    count: int | None = None
    row_ids: list[int] | None = None  # first MAX_LISTED_ROWS only


@dataclass
class NormalizeResult:
    clean: pd.DataFrame  # index = row_id (1-based data row); all canonical columns
    rejected: list[RowIssue]
    warnings: list[Warning_]
    present_fields: set[str] = field(default_factory=set)
    label: pd.Series | None = None  # training only


def snake(header: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(header).strip().lower()).strip("_")


def _clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s == "")


def _numeric(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """-> (values float64, blank mask, invalid mask). Invalid = present but not a finite number."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = s.astype("float64")
        blank = num.isna()
    else:
        cs = _clean_str(s)
        blank = cs.isna()
        num = pd.to_numeric(cs, errors="coerce").astype("float64")
    invalid = ~blank & ~np.isfinite(num)
    return num, blank, invalid


def _category(s: pd.Series) -> pd.Series:
    cs = _clean_str(s).str.lower().str.replace(r"[\s\-/]+", "_", regex=True)
    return cs.astype(object).where(cs.notna(), np.nan)


def resolve_columns(headers: list[str], mapping: dict[str, str] | None, *, training: bool) -> tuple[dict[str, str], list[str]]:
    """canonical field -> source header, plus headers that map to nothing."""
    headers = [str(h).strip() for h in headers]
    if len(set(headers)) != len(headers) or len({snake(h) for h in headers}) != len(headers):
        raise FileValidationError("duplicate_headers", "Duplicate (or indistinguishable) column headers", status=400)
    bad = [h for h in headers if _SENSITIVE_HEADER.search(snake(h))]
    if bad:
        raise FileValidationError(
            "sensitive_column",
            "File contains columns that look like sensitive personal/payment data; remove them and retry",
            {"columns": bad},
            status=400,
        )

    resolved: dict[str, str] = {}
    if mapping:
        unknown = sorted(set(mapping) - set(CANONICAL_NAMES))
        if unknown:
            raise FileValidationError(
                "invalid_mapping", "Mapping refers to unknown canonical fields", {"unknown_fields": unknown, "canonical_fields": list(CANONICAL_NAMES)}, status=400
            )
        absent = sorted({v for v in mapping.values() if str(v).strip() not in headers})
        if absent:
            raise FileValidationError(
                "invalid_mapping", "Mapping refers to columns that are not in the file", {"missing_columns": absent, "file_columns": headers}, status=400
            )
        resolved = {k: str(v).strip() for k, v in mapping.items()}
        if len(set(resolved.values())) != len(resolved):
            raise FileValidationError("invalid_mapping", "Two canonical fields map to the same column", status=400)

    by_snake = {snake(h): h for h in headers}
    taken = set(resolved.values())
    for name in CANONICAL_NAMES:
        if name not in resolved and name in by_snake and by_snake[name] not in taken:
            resolved[name] = by_snake[name]

    required = [f.name for f in FIELDS if f.requirement == REQUIRED or (training and f.name == "label_is_fraud")]
    missing = [n for n in required if n not in resolved]
    if missing:
        raise FileValidationError(
            "missing_required_columns",
            "Required columns are missing; add them or supply a column mapping",
            {"missing": missing, "file_columns": headers},
            status=422,
        )
    unmapped = [h for h in headers if h not in set(resolved.values())]
    return resolved, unmapped


def normalize(raw: pd.DataFrame, mapping: dict[str, str] | None = None, *, training: bool = False) -> NormalizeResult:
    resolved, unmapped = resolve_columns(list(raw.columns), mapping, training=training)
    raw = raw.rename(columns=lambda c: str(c).strip())
    n = len(raw)
    row_ids = pd.RangeIndex(1, n + 1)
    errors: dict[int, list[dict]] = {}
    warnings: list[Warning_] = []

    def reject(mask: pd.Series, fld: str, code: str, message: str) -> None:
        for pos in np.flatnonzero(np.asarray(mask)):
            errors.setdefault(int(pos), []).append({"field": fld, "code": code, "message": message})

    def col(name: str) -> pd.Series | None:
        return raw[resolved[name]].reset_index(drop=True) if name in resolved else None

    def warn_blank(name: str, blank: pd.Series) -> None:
        c = int(np.asarray(blank).sum())
        if c:
            warnings.append(Warning_("missing_values", f"{name}: {c} blank value(s) kept as unknown", [name], c))

    out: dict[str, pd.Series] = {}

    # amount (required)
    s = col("amount")
    amount, blank, invalid = _numeric(s)
    reject(blank, "amount", "missing_required", "amount is required")
    reject(invalid, "amount", "invalid_value", "amount must be a finite number")
    reject(~blank & ~invalid & (amount < 0), "amount", "invalid_value", "amount must be >= 0")
    out["amount"] = amount

    # timestamp (recommended)
    s = col("transaction_timestamp")
    if s is None:
        warnings.append(
            Warning_("missing_column", "transaction_timestamp column not provided; time features are unknown for all rows", ["transaction_timestamp"])
        )
        out["transaction_timestamp"] = pd.Series(pd.NaT, index=range(n), dtype="datetime64[ns, UTC]")
    elif pd.api.types.is_datetime64_any_dtype(s):
        out["transaction_timestamp"] = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    else:
        cs = _clean_str(s)
        blank = cs.isna()
        ts = pd.to_datetime(cs, errors="coerce", utc=True, format="ISO8601")
        reject(~blank & ts.isna(), "transaction_timestamp", "invalid_value", "transaction_timestamp must be ISO-8601")
        naive = int((~blank & ts.notna() & ~cs.fillna("").str.contains(_TZ_SUFFIX)).sum())
        if naive:
            warnings.append(Warning_("naive_timestamps", f"{naive} timestamp(s) had no timezone and were interpreted as UTC", ["transaction_timestamp"], naive))
        warn_blank("transaction_timestamp", blank)
        out["transaction_timestamp"] = ts

    # non-negative numbers / integers
    for name in ("account_age_days", "avg_transaction_amount", "transactions_last_24h"):
        s = col(name)
        if s is None:
            out[name] = pd.Series(np.nan, index=range(n), dtype="float64")
            continue
        num, blank, invalid = _numeric(s)
        reject(invalid, name, "invalid_value", f"{name} must be a finite number")
        ok = ~blank & ~invalid
        reject(ok & (num < 0), name, "invalid_value", f"{name} must be >= 0")
        if FIELD_BY_NAME[name].kind == "integer":
            reject(ok & (num != np.floor(num.fillna(0))), name, "invalid_value", f"{name} must be an integer")
        warn_blank(name, blank)
        out[name] = num

    # categories
    for name in ("merchant_category", "payment_method", "country"):
        s = col(name)
        if s is None:
            out[name] = pd.Series(np.nan, index=range(n), dtype=object)
        else:
            out[name] = _category(s)
            warn_blank(name, out[name].isna())

    # identifiers
    s = col("customer_id")
    out["customer_id"] = _clean_str(s).astype(object) if s is not None else pd.Series(np.nan, index=range(n), dtype=object)
    s = col("transaction_id")
    if s is None:
        out["transaction_id"] = pd.Series([f"row-{i}" for i in row_ids], dtype=object)
    else:
        tid = _clean_str(s)
        gen = tid.isna()
        tid = tid.astype(object)
        tid[gen.values] = [f"row-{i}" for i in row_ids[gen.values]]
        out["transaction_id"] = tid
        dup = tid.duplicated(keep=False)
        if dup.any():
            warnings.append(
                Warning_("duplicate_transaction_id", f"{int(dup.sum())} row(s) share a transaction_id with another row", ["transaction_id"], int(dup.sum()))
            )

    # label
    label = None
    s = col("label_is_fraud")
    if s is not None and not training:
        warnings.append(Warning_("label_column_ignored", "label_is_fraud is present but is never used at inference", ["label_is_fraud"]))
    elif training:
        num, blank, invalid = _numeric(s)
        bad = blank | invalid | ~num.isin([0.0, 1.0])
        reject(bad, "label_is_fraud", "invalid_value", "label_is_fraud must be 0 or 1")
        label = num

    if unmapped:
        warnings.append(Warning_("unmapped_columns_ignored", f"{len(unmapped)} column(s) are not part of the schema and were ignored: {', '.join(unmapped[:20])}", unmapped[:20], len(unmapped)))

    frame = pd.DataFrame({k: v.reset_index(drop=True) for k, v in out.items()})
    frame.index = row_ids
    frame = frame[[c for c in CANONICAL_NAMES if c in frame.columns]]
    keep = np.ones(n, dtype=bool)
    for pos in errors:
        keep[pos] = False
    rejected = [RowIssue(int(row_ids[p]), errs) for p, errs in sorted(errors.items())]
    clean = frame[keep]
    if label is not None:
        label = pd.Series(label.values, index=row_ids)[keep].astype(int)
    present = {k for k in resolved}
    return NormalizeResult(clean=clean, rejected=rejected, warnings=warnings, present_fields=present, label=label)
