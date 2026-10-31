from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    field: str
    code: str
    message: str


class RowResult(BaseModel):
    row_id: int  # 1-based position among the CSV data rows
    transaction_id: str  # from the file, or generated `row-<row_id>`
    prediction: Literal["potentially_suspicious", "not_flagged"]
    risk_score: float  # model-estimated probability; see `explanation.score_interpretation`
    missing_model_inputs: list[str]  # model inputs that were unknown for this row


class RejectedRow(BaseModel):
    row_id: int
    errors: list[Issue]


class WarningOut(BaseModel):
    code: str
    message: str
    fields: list[str] | None = None
    count: int | None = None
    row_ids: list[int] | None = None


class Explanation(BaseModel):
    method: Literal["none"] = "none"  # per-row explanations (e.g. SHAP) are Phase 2
    model_type: str
    features_used: list[str]
    score_interpretation: str
    note: str


class PredictResponse(BaseModel):
    run_id: str | None  # null if the run could not be recorded (see warnings)
    model_version: str
    schema_version: str
    total_rows: int  # data rows in the file
    scored_rows: int
    rejected_row_count: int
    flagged_rows: int
    threshold: float
    results: list[RowResult]
    rejected_rows: list[RejectedRow]  # capped; see rejected_rows_truncated
    rejected_rows_truncated: bool
    warnings: list[WarningOut]
    explanation: Explanation
    disclaimer: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorBody
