import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RESULT_KEYS = {"row_id", "transaction_id", "prediction", "risk_score", "missing_model_inputs"}
TOP_KEYS = {
    "run_id", "model_version", "schema_version", "total_rows", "scored_rows", "rejected_row_count", "flagged_rows", "threshold",
    "results", "rejected_rows", "rejected_rows_truncated", "warnings", "explanation", "disclaimer",
}


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("FRAUDLESS_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FRAUDLESS_MAX_ROWS", "50")
    monkeypatch.setenv("FRAUDLESS_MAX_UPLOAD_BYTES", "20000")
    from fraudless.api.db import engine
    from fraudless.api.main import app

    with TestClient(app, raise_server_exceptions=False) as c:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE audit_runs"))
        yield c


def upload(client, content: bytes | str, name="t.csv", ctype="text/csv", **data):
    if isinstance(content, str):
        content = content.encode()
    return client.post("/predict/file", files={"file": (name, content, ctype)}, data=data)


def test_health_and_model_info(client):
    h = client.get("/health").json()
    assert h == {"status": "ok", "model_loaded": True, "database": "ok"}
    m = client.get("/model-info").json()
    assert m["model_version"] == "test-v0" and m["schema_version"] == "transaction-schema-v1"
    assert m["features"] is None or "label_is_fraud" not in json.dumps(m["features"])
    assert {f["name"] for f in m["supported_schema_fields"]} >= {"amount", "transaction_timestamp", "label_is_fraud"}


def test_predict_sample_file_contract(client):
    r = upload(client, (SAMPLES / "transactions_sample.csv").read_bytes())
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == TOP_KEYS
    assert d["total_rows"] == d["scored_rows"] == 5 and d["rejected_row_count"] == 0
    assert d["flagged_rows"] == sum(x["prediction"] == "potentially_suspicious" for x in d["results"])
    assert [x["row_id"] for x in d["results"]] == [1, 2, 3, 4, 5]
    assert [x["transaction_id"] for x in d["results"]][:2] == ["t-001", "t-002"]
    for x in d["results"]:
        assert set(x) == RESULT_KEYS and 0.0 <= x["risk_score"] <= 1.0
        assert x["prediction"] in {"potentially_suspicious", "not_flagged"}
        assert (x["risk_score"] >= d["threshold"]) == (x["prediction"] == "potentially_suspicious")
    assert d["explanation"]["method"] == "none" and "not" in d["disclaimer"].lower()
    assert d["run_id"]


def test_scores_independent_of_label_column(client):
    base = "amount,payment_method\n5000000,transfer\n10,payment\n"
    with_label = "amount,payment_method,label_is_fraud\n5000000,transfer,0\n10,payment,1\n"
    a, b = upload(client, base).json(), upload(client, with_label).json()
    assert [x["risk_score"] for x in a["results"]] == [x["risk_score"] for x in b["results"]]
    assert any(w["code"] == "label_column_ignored" for w in b["warnings"])


def test_custom_headers_with_mapping_reports_rejected_rows(client):
    mapping = (SAMPLES / "custom_headers.mapping.json").read_text()
    r = upload(client, (SAMPLES / "custom_headers.csv").read_bytes(), mapping=mapping)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total_rows"] == 3 and d["scored_rows"] == 2 and d["rejected_row_count"] == 1
    assert d["rejected_rows"] == [{"row_id": 3, "errors": [{"field": "amount", "code": "invalid_value", "message": "amount must be a finite number"}]}]
    assert [x["transaction_id"] for x in d["results"]] == ["a1", "a2"]  # rejected row is reported, never silently dropped


def test_missing_model_inputs_are_explicit(client):
    d = upload(client, "amount\n100\n200\n").json()
    assert all(x["missing_model_inputs"] == ["transaction_timestamp", "payment_method"] for x in d["results"])
    codes = {w["code"] for w in d["warnings"]}
    assert {"model_input_unavailable", "missing_column"} <= codes


@pytest.mark.parametrize(
    "kwargs,status,code",
    [
        (dict(content=b"", name="e.csv"), 400, "empty_file"),
        (dict(content=b"amount\n", name="e.csv"), 400, "no_data_rows"),
        (dict(content="amount,x\n1\n"), 400, "malformed_csv"),
        (dict(content=b"\x00\x01\x02", name="b.csv"), 400, "not_text"),
        (dict(content="amount\n1\n", name="data.txt", ctype="text/plain"), 415, "unsupported_file_type"),
        (dict(content="amount\n1\n", name="data.csv", ctype="application/pdf"), 415, "unsupported_file_type"),
        (dict(content="payment_method\ncard\n"), 422, "missing_required_columns"),
        (dict(content="amount\nabc\n-1\n"), 422, "no_valid_rows"),
        (dict(content="amount,card_number\n1,4111\n"), 400, "sensitive_column"),
        (dict(content="amount\n1\n", mapping="not json"), 400, "invalid_mapping"),
        (dict(content="amount\n1\n", mapping='{"amount": "nope"}'), 400, "invalid_mapping"),
        (dict(content="amount\n" + "1\n" * 51), 413, "too_many_rows"),
        (dict(content=b"amount\n" + b"1" * 25000), 413, "file_too_large"),
    ],
)
def test_invalid_uploads_get_clear_4xx(client, kwargs, status, code):
    r = upload(client, **kwargs)
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"error"} and body["error"]["code"] == code and body["error"]["message"]
    assert "Traceback" not in r.text


def test_missing_file_part_is_422_not_trace(client):
    r = client.post("/predict/file")
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_internal_errors_do_not_leak_traces(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret internal detail amount=123")

    monkeypatch.setattr(client.app.state.pipeline, "predict_proba", boom, raising=False)
    r = upload(client, "amount\n1\n")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Unexpected server error", "details": {}}}
    assert "secret" not in r.text


def test_runs_store_aggregates_only(client):
    upload(client, "amount,payment_method\n100,transfer\nabc,transfer\n")
    upload(client, "nothing\n1\n")  # rejected file is also recorded
    runs = client.get("/runs").json()
    assert {r["status"] for r in runs} == {"completed", "rejected"}
    done = next(r for r in runs if r["status"] == "completed")
    assert (done["total_rows"], done["scored_rows"], done["rejected_rows"]) == (2, 1, 1)
    from fraudless.api.db import engine

    with engine.connect() as conn:
        cols = {r[0] for r in conn.execute(text("select column_name from information_schema.columns where table_name='audit_runs'"))}
        assert not cols & {"filename", "text", "rows", "data", "results", "scores"}
        assert conn.execute(text("select count(*) from model_versions where model_version='test-v0'")).scalar() == 1
        dump = " ".join(str(v) for row in conn.execute(text("select * from audit_runs")) for v in row)
    assert "transfer" not in dump and "100" not in dump.replace("duration", "")  # no cell values persisted


def test_degrades_visibly_when_db_write_fails(client, monkeypatch):
    import fraudless.api.main as m

    monkeypatch.setattr(m, "SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    d = upload(client, "amount\n1\n").json()
    assert d["run_id"] is None and any(w["code"] == "run_not_recorded" for w in d["warnings"])
