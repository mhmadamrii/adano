import numpy as np
import pandas as pd
import pytest

from fraudless.csvio import check_upload_type, parse_csv
from fraudless.validate import FileValidationError, normalize


def raw(**cols):
    return pd.DataFrame({k: pd.Series(v, dtype="string") for k, v in cols.items()})


def codes(res):
    return {(i.row_id, e["field"], e["code"]) for i in res.rejected for e in i.errors}


def test_happy_path_and_defaults():
    res = normalize(raw(amount=["10", "2.5"], payment_method=["Cash Out", "card-present"]))
    assert list(res.clean.index) == [1, 2]
    assert res.clean.amount.tolist() == [10.0, 2.5]
    assert res.clean.payment_method.tolist() == ["cash_out", "card_present"]
    assert res.clean.transaction_id.tolist() == ["row-1", "row-2"]  # generated, not invented data
    assert not res.rejected
    assert any(w.code == "missing_column" and w.fields == ["transaction_timestamp"] for w in res.warnings)


def test_missing_required_column():
    with pytest.raises(FileValidationError) as e:
        normalize(raw(payment_method=["a"]))
    assert e.value.code == "missing_required_columns" and e.value.details["missing"] == ["amount"]


@pytest.mark.parametrize("value,code", [("abc", "invalid_value"), ("-1", "invalid_value"), ("inf", "invalid_value"), ("nan", "invalid_value"), ("", "missing_required")])
def test_invalid_amount_rows_rejected_not_dropped(value, code):
    res = normalize(raw(amount=["5", value, "7"]))
    assert res.clean.index.tolist() == [1, 3]
    assert codes(res) == {(2, "amount", code)}


def test_bad_integer_and_negative_optionals():
    res = normalize(raw(amount=["1", "1", "1"], transactions_last_24h=["2.5", "-1", "3"], account_age_days=["x", "5", ""]))
    assert codes(res) == {
        (1, "transactions_last_24h", "invalid_value"),
        (2, "transactions_last_24h", "invalid_value"),
        (1, "account_age_days", "invalid_value"),
    }
    assert res.clean.index.tolist() == [3]
    assert np.isnan(res.clean.account_age_days.iloc[0])  # blank stays unknown


def test_timestamp_parsing_and_tz_policy():
    res = normalize(raw(amount=["1", "1", "1", "1"], transaction_timestamp=["2024-03-01T10:00:00Z", "2024-03-01 10:00:00", "2024-03-01T12:00:00+02:00", "yesterday"]))
    assert codes(res) == {(4, "transaction_timestamp", "invalid_value")}
    hours = res.clean.transaction_timestamp.dt.hour.tolist()
    assert hours == [10, 10, 10]  # naive -> UTC, offset converted to UTC
    assert any(w.code == "naive_timestamps" and w.count == 1 for w in res.warnings)


def test_blank_timestamp_is_unknown_with_warning():
    res = normalize(raw(amount=["1", "1"], transaction_timestamp=["2024-03-01T10:00:00Z", ""]))
    assert not res.rejected and res.clean.transaction_timestamp.isna().tolist() == [False, True]
    assert any(w.code == "missing_values" and w.fields == ["transaction_timestamp"] for w in res.warnings)


def test_label_is_ignored_at_inference_and_required_in_training():
    res = normalize(raw(amount=["1"], label_is_fraud=["1"]))
    assert res.label is None and any(w.code == "label_column_ignored" for w in res.warnings)
    with pytest.raises(FileValidationError):
        normalize(raw(amount=["1"]), training=True)
    res = normalize(raw(amount=["1", "2", "3"], label_is_fraud=["0", "1", "2"]), training=True)
    assert codes(res) == {(3, "label_is_fraud", "invalid_value")} and res.label.tolist() == [0, 1]


def test_column_mapping():
    df = raw(Amt=["9"], Method=["TRANSFER"], When=["2024-01-01T00:00:00Z"], extra=["x"])
    res = normalize(df, mapping={"amount": "Amt", "payment_method": "Method", "transaction_timestamp": "When"})
    assert res.clean.amount.tolist() == [9.0] and res.clean.payment_method.tolist() == ["transfer"]
    assert any(w.code == "unmapped_columns_ignored" and w.fields == ["extra"] for w in res.warnings)


@pytest.mark.parametrize(
    "mapping,code",
    [({"nope": "Amt"}, "invalid_mapping"), ({"amount": "Missing"}, "invalid_mapping"), ({"amount": "Amt", "country": "Amt"}, "invalid_mapping")],
)
def test_bad_mapping(mapping, code):
    with pytest.raises(FileValidationError) as e:
        normalize(raw(Amt=["1"]), mapping=mapping)
    assert e.value.code == code


def test_no_header_guessing_without_mapping():
    with pytest.raises(FileValidationError) as e:
        normalize(raw(Amt=["1"]))
    assert e.value.code == "missing_required_columns"


def test_sensitive_and_duplicate_headers_rejected():
    for header in ("card_number", "CVV", "ssn", "Password"):
        with pytest.raises(FileValidationError) as e:
            normalize(raw(amount=["1"], **{header: ["x"]}))
        assert e.value.code == "sensitive_column"
    with pytest.raises(FileValidationError) as e:
        normalize(raw(amount=["1"], Amount_=["2"]))
    assert e.value.code == "duplicate_headers"


def test_duplicate_transaction_ids_warn():
    res = normalize(raw(amount=["1", "1"], transaction_id=["a", "a"]))
    assert any(w.code == "duplicate_transaction_id" for w in res.warnings)


# --- csv parsing / upload checks
def test_parse_ok_bom_and_blank_lines():
    df = parse_csv("﻿amount,payment_method\n1,card\n\n2,\n".encode(), 10)
    assert df.shape == (2, 2) and df.columns.tolist() == ["amount", "payment_method"]


@pytest.mark.parametrize(
    "data,code",
    [
        (b"", "empty_file"),
        (b"   \n", "empty_file"),
        (b"amount\n", "no_data_rows"),
        (b"amount,x\n1\n", "malformed_csv"),
        (b'amount,x\n"1,2\n', "malformed_csv"),
        (b"amount;x\n1;2\n", "unsupported_delimiter"),
        (b"\x00\x01\x02binary", "not_text"),
        ("amount\né".encode("latin-1"), "invalid_encoding"),
    ],
)
def test_parse_rejects(data, code):
    with pytest.raises(FileValidationError) as e:
        parse_csv(data, 10)
    assert e.value.code == code


def test_row_limit():
    with pytest.raises(FileValidationError) as e:
        parse_csv(b"amount\n" + b"1\n" * 11, 10)
    assert e.value.code == "too_many_rows" and e.value.status == 413


@pytest.mark.parametrize("name,ctype", [("data.xlsx", "text/csv"), ("data.txt", "text/plain"), ("data.csv", "application/pdf"), (None, "text/csv")])
def test_unsupported_types(name, ctype):
    with pytest.raises(FileValidationError) as e:
        check_upload_type(name, ctype)
    assert e.value.status == 415


def test_csv_type_ok():
    check_upload_type("Data.CSV", "text/csv; charset=utf-8")
