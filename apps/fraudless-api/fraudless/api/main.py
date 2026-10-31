import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..csvio import check_upload_type, parse_csv
from ..features import MODEL_INPUT_FIELDS, missing_inputs
from ..modeling import load_artifact
from ..schema import FIELDS, SCHEMA_VERSION
from ..validate import FileValidationError, normalize
from .config import Settings
from .db import AuditRun, Base, ModelVersion, SessionLocal, engine
from .schemas import ErrorResponse, Explanation, PredictResponse, RejectedRow, RowResult, WarningOut

log = logging.getLogger("fraudless")

DISCLAIMER = (
    "A flag means 'potentially suspicious' according to a statistical model trained on public, simulated data. "
    "It is not evidence or proof of fraud; do not take adverse action on a flag alone."
)
MAX_REJECTED_IN_RESPONSE = 1000
MAX_WARNING_ROWS = 20


def get_db():
    with SessionLocal() as db:
        yield db


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = Settings.from_env()
    # load once at startup; fail fast with a clear message if the artifact is missing/corrupt
    app.state.pipeline, app.state.meta = load_artifact(app.state.settings.model_dir)
    app.state.db_ok = False
    try:
        Base.metadata.create_all(engine)
        meta = app.state.meta
        with SessionLocal() as db:
            stmt = pg_insert(ModelVersion).values(
                model_version=meta["model_version"],
                schema_version=meta["schema_version"],
                artifact_sha256=meta["artifact_sha256"],
                trained_at=meta.get("trained_at", ""),
                model_metadata=meta,
            )
            db.execute(stmt.on_conflict_do_update(index_elements=["model_version"], set_={"artifact_sha256": stmt.excluded.artifact_sha256, "model_metadata": stmt.excluded.model_metadata, "schema_version": stmt.excluded.schema_version, "trained_at": stmt.excluded.trained_at}))
            db.commit()
        app.state.db_ok = True
    except Exception as e:  # service still scores, but says so in /health and in each response
        log.error("database unavailable at startup: %s", type(e).__name__)
    yield


app = FastAPI(title="fraudless-api", version="0.1.0", lifespan=lifespan)


# ---- errors: one stable shape, never a raw trace
def _err(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "details": details or {}}})


@app.exception_handler(FileValidationError)
async def _file_err(_: Request, e: FileValidationError):
    return _err(e.status, e.code, e.message, e.details)


@app.exception_handler(RequestValidationError)
async def _req_err(_: Request, e: RequestValidationError):
    fields = [".".join(str(p) for p in err["loc"]) for err in e.errors()]
    return _err(422, "invalid_request", "The request is missing or has invalid parameters", {"fields": fields})


@app.exception_handler(StarletteHTTPException)
async def _http_err(_: Request, e: StarletteHTTPException):
    return _err(e.status_code, "http_error", str(e.detail))


@app.exception_handler(Exception)
async def _unhandled(_: Request, e: Exception):
    log.error("unhandled error: %s", type(e).__name__)  # type only: tracebacks can carry row data
    return _err(500, "internal_error", "Unexpected server error")


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    limit = request.app.state.settings.max_upload_bytes
    cl = request.headers.get("content-length")
    if request.url.path == "/predict/file" and cl and cl.isdigit() and int(cl) > limit + 64 * 1024:
        return _err(413, "file_too_large", f"The upload exceeds the {limit} byte limit", {"max_bytes": limit})
    return await call_next(request)


# ---- run recording (aggregates only)
def record_run(**fields) -> str | None:
    try:
        with SessionLocal() as db:
            run = AuditRun(**fields)
            db.add(run)
            db.commit()
            return str(run.id)
    except Exception as e:
        log.error("could not record audit run: %s", type(e).__name__)
        return None


def parse_mapping(raw: str | None) -> dict[str, str] | None:
    if raw is None or not raw.strip():
        return None
    try:
        mapping = json.loads(raw)
    except json.JSONDecodeError:
        raise FileValidationError("invalid_mapping", "mapping must be a JSON object like {\"amount\": \"Amt\"}", status=400)
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()):
        raise FileValidationError("invalid_mapping", "mapping must be a JSON object of canonical field -> column name strings", status=400)
    return mapping


# ---- endpoints
@app.get("/health")
def health():
    ok = False
    try:
        with SessionLocal() as db:
            db.execute(text("select 1"))
        ok = True
    except Exception:
        pass
    return {"status": "ok" if ok else "degraded", "model_loaded": True, "database": "ok" if ok else "unavailable"}


@app.get("/model-info")
def model_info(request: Request):
    m = request.app.state.meta
    return {
        "model_version": m["model_version"],
        "schema_version": m["schema_version"],
        "model_type": m["model_type"],
        "trained_at": m.get("trained_at"),
        "threshold": m["threshold"],
        "threshold_strategy": m.get("threshold_strategy"),
        "features": m.get("features"),
        "metrics": m.get("metrics"),
        "dataset": {k: m.get("dataset", {}).get(k) for k in ("name", "source_url", "license", "label_definition")},
        "supported_schema_fields": [{"name": f.name, "type": f.kind, "requirement": f.requirement, "description": f.description} for f in FIELDS],
        "limitations": "Trained on simulated (PaySim-style) data using only amount, time-of-day and payment method; public benchmark performance does not predict performance on a company's real transactions.",
    }


@app.get("/runs")
def list_runs(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.scalars(select(AuditRun).order_by(AuditRun.created_at.desc()).limit(min(max(limit, 1), 200)))
    return [
        {
            "run_id": str(r.id), "created_at": r.created_at.isoformat(), "status": r.status, "error_code": r.error_code,
            "model_version": r.model_version, "total_rows": r.total_rows, "scored_rows": r.scored_rows,
            "rejected_rows": r.rejected_rows, "flagged_rows": r.flagged_rows, "warning_codes": r.warning_codes, "duration_ms": r.duration_ms,
        }
        for r in rows
    ]


@app.post("/predict/file", response_model=PredictResponse, responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict_file(request: Request, file: UploadFile = File(...), mapping: str | None = Form(None)):
    st, meta, pipeline = request.app.state.settings, request.app.state.meta, request.app.state.pipeline
    started = time.perf_counter()
    # Processed in memory (bounded by the size limit); no upload is written to disk or kept.
    data = file.file.read(st.max_upload_bytes + 1)
    base = dict(model_version=meta["model_version"], schema_version=SCHEMA_VERSION, file_sha256=hashlib.sha256(data).hexdigest(), file_size_bytes=len(data))
    ms = lambda: int((time.perf_counter() - started) * 1000)  # noqa: E731

    try:
        if len(data) > st.max_upload_bytes:
            raise FileValidationError("file_too_large", f"The upload exceeds the {st.max_upload_bytes} byte limit", {"max_bytes": st.max_upload_bytes}, status=413)
        check_upload_type(file.filename, file.content_type)
        raw = parse_csv(data, st.max_rows)
        res = normalize(raw, parse_mapping(mapping))
        if res.clean.empty:
            raise FileValidationError(
                "no_valid_rows", "Every row failed validation; nothing was scored",
                {"rejected_row_count": len(res.rejected), "rejected_rows": [{"row_id": i.row_id, "errors": i.errors} for i in res.rejected[:50]]},
            )
    except FileValidationError as e:
        record_run(status="rejected", error_code=e.code, warning_codes=[], duration_ms=ms(), **base)
        raise

    proba = pipeline.predict_proba(res.clean)[:, 1]
    flagged = proba >= meta["threshold"]
    missing = missing_inputs(res.clean)

    warnings = [WarningOut(code=w.code, message=w.message, fields=w.fields, count=w.count, row_ids=w.row_ids) for w in res.warnings]
    for f in MODEL_INPUT_FIELDS:
        if f != "amount" and f not in res.present_fields:
            warnings.append(WarningOut(code="model_input_unavailable", message=f"{f} was not provided; the model treated it as unknown for every row (unknown is not treated as legitimate)", fields=[f]))

    results = [
        RowResult(
            row_id=int(rid), transaction_id=str(tid),
            prediction="potentially_suspicious" if fl else "not_flagged",
            risk_score=round(float(p), 6), missing_model_inputs=list(mi),
        )
        for rid, tid, fl, p, mi in zip(res.clean.index, res.clean["transaction_id"], flagged, proba, missing)
    ]
    n_flagged, total = int(flagged.sum()), len(raw)
    run_id = record_run(
        status="completed", error_code=None, total_rows=total, scored_rows=len(results), rejected_rows=len(res.rejected),
        flagged_rows=n_flagged, warning_codes=sorted({w.code for w in warnings}), duration_ms=ms(), **base,
    )
    if run_id is None:
        warnings.append(WarningOut(code="run_not_recorded", message="The audit run could not be saved to the database; results are still valid"))

    return PredictResponse(
        run_id=run_id, model_version=meta["model_version"], schema_version=SCHEMA_VERSION,
        total_rows=total, scored_rows=len(results), rejected_row_count=len(res.rejected), flagged_rows=n_flagged,
        threshold=meta["threshold"], results=results,
        rejected_rows=[RejectedRow(row_id=i.row_id, errors=i.errors) for i in res.rejected[:MAX_REJECTED_IN_RESPONSE]],
        rejected_rows_truncated=len(res.rejected) > MAX_REJECTED_IN_RESPONSE,
        warnings=warnings,
        explanation=Explanation(
            model_type=meta["model_type"], features_used=list(meta["features"]["canonical_inputs"]),
            score_interpretation=meta.get("risk_score_interpretation", ""),
            note="No per-row explanation in Phase 1. missing_model_inputs lists inputs the model had to treat as unknown for that row.",
        ),
        disclaimer=DISCLAIMER,
    )
