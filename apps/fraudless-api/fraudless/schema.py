"""Versioned canonical transaction schema (the data contract). See docs/DATA_CONTRACT.md."""
from dataclasses import dataclass

SCHEMA_VERSION = "transaction-schema-v1"

REQUIRED = "required"
RECOMMENDED = "recommended"
OPTIONAL = "optional"
TRAINING_ONLY = "training_only"


@dataclass(frozen=True)
class Field:
    name: str
    kind: str  # string | category | number | integer | datetime | label
    requirement: str
    description: str


FIELDS: tuple[Field, ...] = (
    Field("transaction_id", "string", OPTIONAL, "Row identifier; an internal `row-<n>` id is generated if absent"),
    Field("amount", "number", REQUIRED, "Finite, non-negative"),
    Field("transaction_timestamp", "datetime", RECOMMENDED, "ISO-8601; naive values are interpreted as UTC"),
    Field("merchant_category", "category", OPTIONAL, "Normalized: trimmed, lower-case, spaces/hyphens -> underscore"),
    Field("payment_method", "category", OPTIONAL, "Normalized like merchant_category"),
    Field("country", "category", OPTIONAL, "Normalized like merchant_category (no ISO mapping in v1)"),
    Field("customer_id", "string", OPTIONAL, "Pseudonymous/tokenized id; never a direct identity"),
    Field("account_age_days", "number", OPTIONAL, "Finite, non-negative"),
    Field("transactions_last_24h", "integer", OPTIONAL, "Non-negative integer"),
    Field("avg_transaction_amount", "number", OPTIONAL, "Finite, non-negative"),
    Field("label_is_fraud", "label", TRAINING_ONLY, "0 = legitimate, 1 = confirmed fraud; never used at inference"),
)

FIELD_BY_NAME = {f.name: f for f in FIELDS}
CANONICAL_NAMES = tuple(f.name for f in FIELDS)
